for count in range(100):

    fout = open("train_test_data/block_" + str(count) + ".txt", "w")
    filename = "train_data/min" + str(count) + ".txt"

    with open(filename,encoding="utf8") as f:

        for line in f:

            splitLine = line.split()

            if(splitLine[0]=='Macros'):
                macros = True
                print(line, end='', file=fout)
                continue
            if(splitLine[0] == 'Edges'):
                macros = False
                print(line, end='', file=fout)

                continue

            if(macros):
                index = int(splitLine[0])
                XL = int(splitLine[1])
                YL = int(splitLine[2])
                XR = int(splitLine[3])
                YR = int(splitLine[4])
                xdim = XR - XL + 1
                ydim = YR - YL + 1
                print(index, xdim, ydim, file=fout)

            else:
                print(line, end='', file=fout)

    fout.close()

