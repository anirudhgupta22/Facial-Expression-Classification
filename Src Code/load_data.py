import numpy as np

# auxillary function to load the data
def faces_load_data():
 
    skip_rows = 1
    train_size = 28709
    test_size = 3589
    #train_size = 2000
    #test_size = 0
    dim = 48
    X_train = np.empty([train_size,dim, dim])
    X_test = np.empty([test_size, dim, dim])
    y_train = np.empty(train_size)
    y_test = np.empty(test_size)
    
    f = open('fer2013.csv', 'r')
    lines = f.readlines()
    train_index = test_index = 0
    print len(lines)
    for i in range(0,0):
        
        line = lines[i]
        if i >= skip_rows:
            split_line = line.split(",")
            usage = split_line[2].rstrip()
            if usage == 'Training':
                X_train[train_index, :,:] = np.fromstring(split_line[1], dtype = 'int', sep = ' ').reshape(dim, dim)
                y_train[train_index] = int(split_line[0])
                train_index += 1
            elif usage == 'PublicTest':
                X_test[test_index, :,:] = np.fromstring(split_line[1], dtype = 'int', sep = ' ').reshape(dim, dim)
                y_test[test_index] = int(split_line[0])
                test_index += 1
                 
    return (X_train, y_train) , (X_test, y_test)



faces_load_data()
