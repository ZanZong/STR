import sys
import time
import contextlib

######################
# Method1 
# class MyResource1:
#     def __enter__(self):
#         print('connect to resource')
#         return self    # return可以是对象，然后会绑定到as后面的变量
 
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         print('close resource connection')
#         # __exit__方法 
#         # return True 的话，如果执行 query 有异常，则异常就不往上抛了
# 		# return False 的话，如果执行 query 有异常，则异常就继续往上抛了
 
#     def query(self):
#         print('query data')
 
 
# with MyResource1() as r:
#     r.query()

# # Method2
# class MyResource2:
 
#     def query(self):
#         print('query data')
 
# # contextmanager 简化上下文管理器复杂的定义
# from contextlib import contextmanager
 
# @contextmanager
# def make_myresource():
#     print('connect to resource')
#     yield MyResource2()     # yield 相当于return 加 中断回执。yield后面的语句会最终回来继续执行
#     print('close resource connection')

# with make_myresource() as r:
#     r.query()

####################

ops = list()

@contextlib.contextmanager
def capture_ops():
  """Decorator to capture ops created in the block.
  with capture_ops() as ops:
    # create some ops
  print(ops) # => prints ops created.
  """
  op_list = []

  yield op_list

  op_list.extend(filter(lambda x: str(x).startswith("_"), ops))


ops.append("1")
ops.append("2")
with capture_ops() as cap:
    ops.append("_1")
    ops.append("_2")
    ops.append("_3")

print(cap)