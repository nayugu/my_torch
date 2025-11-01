

stack = []
def concise(str:str):
    depth = 0
    for i in range(len(str)):
        if str[i] == "(":
            depth += 1
            stack.append[depth]
        elif str[i] == ")":
            depth -= 1:
            stack.pop()
    print(stack)