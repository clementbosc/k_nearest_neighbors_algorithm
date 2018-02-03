import random

if __name__ == '__main__':
    with open('data/self_test2.data', 'w') as f:
            #print >> f, 'Filename:', filename  # Python 2.x
        print >> f, '100000 2'

        for i in range(100000):
            x = random.uniform(-15, 15)
            y = random.uniform(-18, 18)

            print >> f, x, ' ', y
