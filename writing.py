class WriteData:
    def WritingInFile(self, names, sequences, fileName):
        with open(fileName, 'w') as f:
            for i in range(len(names)):
                f.write(names[i] + ':\n')
                for j in range(len(sequences[i])):
                    f.write('\t' + str(sequences[i][j]) + '\n')

    def WritingInFile1(self, names, sequences, fileName):
        with open(fileName, "w") as file:
            for line in sequences:
                print(line, file=file)