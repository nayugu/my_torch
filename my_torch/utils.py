
def concise(str:str):
    max_depth = 3
    depth = 0
    content = ""
    #stack = []
    for i in range(len(str)):
        if str[i] == "(":
            depth += 1
            #stack.append(depth)

            if depth == max_depth + 1:
                content += "?"

        elif str[i] == ")":
            depth -= 1
            #stack.pop()

        if depth <= max_depth:
            content += str[i]

        #print(stack,10*" ",str[:i+1],10*" ",content)
    return content