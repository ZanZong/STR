import itertools
from stropt.core.utils import swapping
from typing import List, Dict, Tuple, Optional

import numpy as np

from stropt.core.dfgraph import DFGraph
from stropt.core.schedule import OperatorEvaluation, AllocateRegister, DeallocateRegister, Schedule, SchedulerAuxData
from stropt.core.utils.timer import Timer
from stropt.core.utils.swapping import SwapControler

# Simulate DNN execution to get runtime memory information
class ScheduleBuilder:
    def __init__(self, g, verbosity: int = 2):
        self.max_ram = 0
        self.total_cpu = 0
        self.g = g
        self.schedule: Schedule = []
        self.live_registers: Dict[int, int] = {}
        self.next_free_register_id = 0
        self.verbosity = verbosity
        self.ram_timeline: List[int] = []
        self.unfinished_swap_in_ops_idx = []

    def is_op_cached(self, op_id: int):
        return op_id in self.live_registers.keys()

    def allocate_register(self, op_id: int):
        """
        Schedule a register allocation
        :param op_id: ID for operation whose output will be stored in this register,
        :return: the newly allocated register ID
        """
        if op_id in self.live_registers.keys():
            if self.verbosity >= 2:
                print("WARNING! Double allocating output register for op #{}, skipping allocation to reuse reg #{}"
                      .format(op_id, self.live_registers[op_id]))
            return self.live_registers[op_id]
        reg = AllocateRegister(self.next_free_register_id, op_id, self.g.cost_ram[op_id])
        self.live_registers[op_id] = reg.register_id
        self.schedule.append(reg)
        self.next_free_register_id += 1
        self.max_ram = max(self.max_ram, self.current_mem())
        self.ram_timeline.append(self.current_mem())
        return reg.register_id

    def run_operator(self, op_id: int, update_aux_vars: bool):
        debug_str = "Dependency not fulfilled for op #{}, ops in ram now are {} but I need {}".format(
            op_id, set(self.live_registers.keys()), self.g.predecessors(op_id))
        assert all([pred in self.live_registers.keys() for pred in self.g.predecessors(op_id)]), debug_str
        out_reg = self.allocate_register(op_id)
        in_regs = {pred_id: self.live_registers[pred_id] for pred_id in self.g.predecessors(op_id)}
        eval_op = OperatorEvaluation(op_id, in_regs, out_reg, self.g.cost_cpu[op_id],
                                    update_aux_vars=update_aux_vars, is_backwards=op_id > self.g.vloss)
                                    #  update_aux_vars=update_aux_vars, is_backwards=op_id not in self.g.vfwd)
        self.schedule.append(eval_op)
        self.total_cpu += self.g.cost_cpu[op_id]
        self.ram_timeline.append(self.current_mem())

    def deallocate_register(self, op_id: int):
        """
        Schedule a register deallocation
        :param op_id: ID for operation whose output will be stored in this register
        """
        if op_id not in self.live_registers.keys():
            print("WARNING! Double free output register for op #{}".format(op_id))
        reg_id = self.live_registers.pop(op_id)
        self.schedule.append(DeallocateRegister(op_id, reg_id))
        self.ram_timeline.append(self.current_mem())

    def current_mem(self):
        # unfinished swap in operators are still unregistered
        return sum(map(self.g.cost_ram.get, set(self.live_registers.keys()).union(set(self.unfinished_swap_in_ops_idx))))

    def start_swap_in(self, index: int):
        """Memory consumed but tensor is not ready"""
        self.unfinished_swap_in_ops_idx.append(index)

    def finish_swap_in(self, index: int):
        """Tensor is ready to use"""
        self.allocate_register(index)
        self.unfinished_swap_in_ops_idx.remove(index)

def schedule_from_rs(g: DFGraph, r: np.ndarray, s: np.ndarray) -> Tuple[Optional[Schedule], Optional[SchedulerAuxData]]:
    if r is None or s is None:
        return None, None  # infeasible
    T = g.size

    def _used_after(t_, u_, i_):
        """Returns True if v_u is used after v_i in stage t"""
        is_retained_snapshot = t_ < T - 1 and s[t_ + 1, u_] == 1
        is_used_by_successor = not all([r[t_, v] == 0 or v <= i_ for v in g.successors(u_)])
        return is_retained_snapshot or is_used_by_successor

    with Timer('schedule_rs_matrix') as schedule_timer:
        # compute last usage to determine whether to update auxiliary variables
        last_used = {i: max([t for t in range(T) if r[t, i] == 1]) for i in range(T)}
        mem_usage = np.zeros((T, T), dtype=np.int)
        sb = ScheduleBuilder(g, verbosity=1)
        stage_memory_timeline = [] # only record peak memory of each stage
        last_timeline_idx = 0
        for t in range(T):
            # Free unused checkpoints
            for i in filter(lambda x: sb.is_op_cached(x), range(T)):
                if not _used_after(t, i, i):
                    sb.deallocate_register(i)

            for i in range(T):
                if r[t, i] == 1:
                    sb.run_operator(i, last_used[i] == t)
                mem_usage[t, i] = sb.current_mem() + g.cost_ram_fixed

                # Free memory
                for u in filter(lambda x: sb.is_op_cached(x), itertools.chain(g.predecessors(i), [i])):
                    if not _used_after(t, u, i):
                        sb.deallocate_register(u)
            stage_memory_timeline.append(max(sb.ram_timeline[last_timeline_idx:]))
            last_timeline_idx = len(sb.ram_timeline)
        total_ram = sb.max_ram + g.cost_ram_fixed
        ram_timeline = [mem + g.cost_ram_fixed for mem in sb.ram_timeline]

    return sb.schedule, SchedulerAuxData(R=r, S=s, cpu=sb.total_cpu, peak_ram=total_ram,
                                         activation_ram=sb.max_ram, mem_grid=mem_usage,
                                         mem_timeline=stage_memory_timeline, schedule_time_s=schedule_timer.elapsed)


def schedule_from_rspq(g: DFGraph, r: np.ndarray, s: np.ndarray, q: np.ndarray) -> Tuple[Optional[Schedule], Optional[SchedulerAuxData]]:
    if r is None or s is None or q is None:
        return None, None  # infeasible
    T = g.size
    swap_ctrl = SwapControler(10E9, g.cost_cpu, g.cost_ram)

    def _used_after(t_, u_, i_):
        """Returns True if v_u is used after v_i in stage t"""
        finish_stage = swap_ctrl.swap_finish_stage(t_, u_)
        if finish_stage is not None and finish_stage < T - 1:
            is_retained_snapshot = (t_ < T - 1 and s[t_ + 1, u_] == 1) or s[finish_stage + 1, u_] == 1 # checkpointed or swapped in row axis
        else:
            is_retained_snapshot = (t_ < T - 1 and s[t_ + 1, u_] == 1)
        is_used_by_successor = not all([r[t_, v] == 0 or v <= i_ for v in g.successors(u_)]) # used in column axis
        # if not (is_retained_snapshot or is_used_by_successor):
        #     print(f"Tensor {u_} is freed at stage {t_} after compute {i} because:" + \
        #                 f"is_retained_snapshot={is_retained_snapshot}, is_used_by_successor={is_used_by_successor}")
        return is_retained_snapshot or is_used_by_successor

    with Timer('schedule_rspq_matrix') as schedule_timer:
        # compute last usage to determine whether to update auxiliary variables
        last_used = {i: max([t for t in range(T) if r[t, i] == 1]) for i in range(T)}
        mem_usage = np.zeros((T, T), dtype=np.int)
        sb = ScheduleBuilder(g, verbosity=1)

        # find all swapping ops
        swap_in_ops = dict()
        for t in range(T):
            for i in range(T):
                if q[t, i] == 1:
                    if t in swap_in_ops.keys():
                        print(f"Error: More than 1 tensor start swap in at stage {t}: {swap_in_ops.keys()} + {i}")
                    else:
                        swap_in_ops[t] = i
                    break
        
        finish_swap_in_stage = None
        current_swap_in_idx = None

        stage_memory_timeline = [] # only record peak memory of each stage
        last_timeline_idx = 0
        for t in range(T):
            # Free unused checkpoints
            for i in filter(lambda x: sb.is_op_cached(x), range(T)):
                if not _used_after(t, i, i):
                    sb.deallocate_register(i)
            # register swap-in tensors
            if t in swap_in_ops.keys():
                assert finish_swap_in_stage is None, f"Previous tensor has not finish the swap in"
                current_swap_in_idx = swap_in_ops[t]
                sb.start_swap_in(current_swap_in_idx)
                finish_swap_in_stage = swap_ctrl.swap_finish_stage(t, current_swap_in_idx)
                # print(f"Allocate tensor {swap_in_ops[t]} at stage {t}, expect finish at {finish_swap_in_stage}.")
            if current_swap_in_idx is not None and t == finish_swap_in_stage:
                sb.finish_swap_in(current_swap_in_idx)
                # print(f"Register tensor {current_swap_in_idx} at stage {t} by swap in.")
                finish_swap_in_stage = None
                current_swap_in_idx = None

            for i in range(T):
                if r[t, i] == 1:
                    sb.run_operator(i, last_used[i] == t)
                mem_usage[t, i] = sb.current_mem() + g.cost_ram_fixed

                # Free memory
                for u in filter(lambda x: sb.is_op_cached(x), itertools.chain(g.predecessors(i), [i])):
                    if not _used_after(t, u, i):
                        sb.deallocate_register(u)
            stage_memory_timeline.append(max(sb.ram_timeline[last_timeline_idx:]))
            last_timeline_idx = len(sb.ram_timeline)

        total_ram = sb.max_ram + g.cost_ram_fixed
        ram_timeline = [mem + g.cost_ram_fixed for mem in sb.ram_timeline]

    return sb.schedule, SchedulerAuxData(R=r, S=s, cpu=sb.total_cpu, peak_ram=total_ram,
                                         activation_ram=sb.max_ram, mem_grid=mem_usage,
                                         mem_timeline=stage_memory_timeline, schedule_time_s=schedule_timer.elapsed)