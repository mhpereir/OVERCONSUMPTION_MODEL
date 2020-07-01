# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:09:36 2020

@author: Matthew Wilson
"""



from multiprocessing import Lock, Process, Queue, current_process
import numpy as np
from time import time
import queue # imported for using queue.Empty exception

def f(arr):
    return np.sqrt( np.power(10, np.log10(np.exp(np.log(arr)))) ** 2)

n = 1e8
arr = np.arange(1,n)

def main():
    number_of_task = 6
    tasks_to_accomplish = Queue()
    processes = []
    
    _start = time()

    for i in range(number_of_task):
        dn = np.floor(n/number_of_task)
        tasks_to_accomplish.put(f(arr[int(dn*i):int(dn*(i+1))]))
    
    print('Time to queue: ', time() - _start)
    
        
    # creating processes
    
    for w in range(number_of_task):
        out = tasks_to_accomplish.get()
        print(out)
    
    print('Total time: ', time() - _start)


if __name__ == '__main__':
    main()