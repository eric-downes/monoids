import time
import os
import random
from multiprocessing import Process, Queue, Lock
 
 
# Producer function that places data on the Queue
def producer(queue, lock, names):
    # Synchronize access to the console
    with lock:
        print('Starting producer => {}'.format(os.getpid()))
         
    # Place our names on the Queue
    for name in names:
        time.sleep(random.randint(0, 10))
        queue.put(name)
 
    # Synchronize access to the console
    with lock:
        print('Producer {} exiting...'.format(os.getpid()))
 
 
# The consumer function takes data off of the Queue
def consumer(queue, lock):
    # Synchronize access to the console
    with lock:
        print('Starting consumer => {}'.format(os.getpid()))
     
    # Run indefinitely
    while True:
        time.sleep(random.randint(0, 10))
         
        # If the queue is empty, queue.get() will block until the queue has data
        name = queue.get()
 
        # Synchronize access to the console
        with lock:
            print('{} got {}'.format(os.getpid(), name))
 
 
if __name__ == '__main__':
     
    # Some lists with our favorite characters
    names = [['Master Shake', 'Meatwad', 'Frylock', 'Carl'],
             ['Early', 'Rusty', 'Sheriff', 'Granny', 'Lil'],
             ['Rick', 'Morty', 'Jerry', 'Summer', 'Beth']]
 
    # Create the Queue object
    queue = Queue()
     
    # Create a lock object to synchronize resource access
    lock = Lock()
 
    producers = []
    consumers = []
 
    for n in names:
        # Create our producer processes by passing the producer function and it's arguments
        producers.append(Process(target=producer, args=(queue, lock, n)))
 
    # Create consumer processes
    for i in range(len(names) * 2):
        p = Process(target=consumer, args=(queue, lock))
         
        # This is critical! The consumer function has an infinite loop
        # Which means it will never exit unless we set daemon to true
        p.daemon = True
        consumers.append(p)
 
    # Start the producers and consumer
    # The Python VM will launch new independent processes for each Process object
    for p in producers:
        p.start()
 
    for c in consumers:
        c.start()
 
    # Like threading, we have a join() method that synchronizes our program
    for p in producers:
        p.join()
 
    print('Parent process exiting...')
    
