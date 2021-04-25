import multiprocessing as mp
import random
import time

def producer(input_queue, output_queue):
    while True:
        work_item = input_queue.get()
        if work_item is None:
            # notify the consumers that one of the producers exited
            output_queue.put(None)
            print('Producer exiting')
            return None
        for item in work_item:
            output_queue.put(item)


def consumer(output_queue, producers_exited):
    # probably best to use a while True loop explicitly
    # look for sentinel value and return
    while True:
        with producers_exited.get_lock():
            # if no more input expected and the queue is empty, exit process
            if producers_exited.value == 16:
                try:
                    work_item = output_queue.get(timeout=1)
                except:
                    print('Consumer exiting')
                    return None
            else:
                work_item = output_queue.get()
        
                if work_item is None:
                    # a producer process exited
                    producers_exited.value += 1
                    continue

                print(work_item)


if __name__ == '__main__':
    n_producers = 16
    input_queue = mp.Queue()
    # add some random lists to the input queue
    num_lists = 100
    for _ in range(num_lists):
        rand_length = random.randrange(start=100, stop=200)
        input_queue.put(list(range(rand_length)))

    for _ in range(n_producers):
        # each worker should read one sentinel value and exit
        input_queue.put(None)

    n_consumers = 16
    producers_exited = mp.Value('i', 0)
    output_queue = mp.Queue(maxsize=50)

    # start producer processes
    producer_processes = []
    for i in range(n_producers):
        p = mp.Process(target=producer, args=(input_queue, output_queue))
        p.start()
        p.name = f'producer-process-{i}'
        producer_processes.append(p)
    
    time.sleep(2)
    
    # start consumer processes
    consumer_processes = []
    for i in range(n_consumers):
        p = mp.Process(target=consumer, args=(output_queue, producers_exited))
        p.start()
        p.name = f'consumer-process-{i}'
        consumer_processes.append(p)
    
    for p in producer_processes:
        p.join()
    
    for p in consumer_processes:
        p.join()

    input_queue.close()
    output_queue.close()

