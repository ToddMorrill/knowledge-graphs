import multiprocessing as mp
import random

def producer(input_queue, output_queue):
    while True:
        work_item = input_queue.get()
        if work_item is None:
            # pass this information along to the other producers
            input_queue.put(None)
            
            # notify the consumers that one of the producers exited
            output_queue.put(None)
            print('Producer exiting')
            return None
        for item in work_item:
            output_queue.put(item)


def consumer(output_queue, final_queue, producers_exited):
    while True:
        with producers_exited.get_lock():
            # if no more input expected and the queue is empty, exit process
            if producers_exited.value == 16:
                if output_queue.empty():
                    print('Consumer exiting')
                    return None
            else:
                work_item = output_queue.get()
        
                if work_item is None:
                    # a producer process exited
                    producers_exited.value += 1
                    continue

        # do work, NB: must be outside with context to get true parallelism
        final_queue.put(work_item)


if __name__ == '__main__':
    n_producers = 16
    input_queue = mp.Queue()
    # add some random lists to the input queue
    num_lists = 100
    expected = 0
    for _ in range(num_lists):
        rand_length = random.randrange(start=100, stop=200)
        input_queue.put(list(range(rand_length)))
        expected += rand_length

    # add sentinel value to the queue (to be shared among the processes)
    # when the producer sees the sentinel value, it should exit
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
    
    # simply used to verify everything finished
    final_queue = mp.Queue()

    # start consumer processes
    consumer_processes = []
    for i in range(n_consumers):
        p = mp.Process(target=consumer, args=(output_queue, final_queue, producers_exited))
        p.start()
        p.name = f'consumer-process-{i}'
        consumer_processes.append(p)
    
    for p in producer_processes:
        p.join()

    # must inspect this before trying to join the consumer process o/w main process will hang
    # https://stackoverflow.com/questions/56321756/multiprocessing-queue-with-hugh-data-causes-wait-for-tstate-lock
    # https://docs.python.org/3.5/library/multiprocessing.html#pipes-and-queues
    results = []
    while not final_queue.empty():
        results.append(final_queue.get())
    assert expected == len(results)

    for p in consumer_processes:
        p.join()

    input_queue.close()
    output_queue.close()
    final_queue.close()

