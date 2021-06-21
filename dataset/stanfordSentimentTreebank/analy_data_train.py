# ==== 分析最短的句子长度 ====

with open("train.txt", 'r') as f:
    lines = f.readlines()
    counter = len(lines[0].strip().split())
    for line in lines:
        line = line.strip().split()
        if len(line) < counter:
            counter = len(line)
    f.close()
print(counter)