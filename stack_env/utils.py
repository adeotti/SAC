from tqdm import tqdm

def print_queue_loading(queue): # tracking queue size mainly during warmup phase
    pbar = tqdm(total=queue._maxsize,desc="Warmup")
    while True:
        pbar.n = queue.qsize()
        pbar.refresh()
        time.sleep(0.1)
        if queue.qsize() == queue._maxsize:
            break
    pbar.n = queue.qsize()
    pbar.refresh()
    pbar.close()


