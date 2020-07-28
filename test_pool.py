from multiprocessing import Pool
import time

def square(x):
    time.sleep(0.01)
    return x**2

inputs = []
normal = []


for i in range(100):
    inputs.append(i)

start_normal = time.time()
for i in range(100):
    normal.append(square(i))
print("normal time: " +str(time.time()-start_normal))

start_normal = time.time()
pool = Pool()
parallel = pool.map(square, inputs)
print("pool time: " +str(time.time()-start_normal))


print(normal==parallel)
print(parallel)
