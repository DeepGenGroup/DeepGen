

filename = "/home/xushilong/CodeGenDemo/Runtime/python/kcg/temp.mlir"
to_analyze_lines = []
with open(filename) as f:
    lines = f.readlines()
    for line in lines :
        if line.find("memref") > 0:
            to_analyze_lines.append(line)
exprs = []
for e in to_analyze_lines :
    st = e.find("[")
    if st < 0:
        continue
    et = e.find("]")
    if et < 0:
        continue
    exprs.append(e[st+1:et])
    print(e[st+1:et])
    print(e)
