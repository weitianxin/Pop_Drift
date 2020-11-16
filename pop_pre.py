import numpy as np
root = './data/ml_10m/'
item_list = []
for i in range(13):
    path = root+'t_{}.txt'.format(i)
    with open(path) as f:
        for line in f:
            item_list.append(int(line.split()[0]))

n_item = len(set(item_list))
pop_item = []
for i in range(13):
    path = root+'t_{}.txt'.format(i)
    total = 0
    item_pop_list_t=[]
    with open(path) as f:
        for line in f:
            line = line.strip().split()
            item, pop = int(line[0]), len(line[1:])
            item_pop_list_t.append((item,pop))
            total+=pop
    pop_item.append([1/(total+n_item) for _ in range(n_item)])
    # pop_item.append([0/(total) for _ in range(n_item)])
    for item,pop in item_pop_list_t:
        pop_item[i][item] = (pop+1)/(total+n_item)
        # pop_item[i][item] = 1e6*(pop)/(total)
pop_item = np.array(pop_item)
# 0-1
pop_item = (pop_item-np.min(pop_item))/(np.max(pop_item)-np.min(pop_item))
with open(root+"item_pop_seq_ori.txt","w") as f:
    for i in range(n_item):
        pop_seq_i = pop_item[:, i]
        write_str = ""
        write_str += str(i) + ' '
        for pop in pop_seq_i:
            write_str += str(pop) + ' '
        write_str = write_str.strip(' ')
        write_str += '\n'
        f.write(write_str)
