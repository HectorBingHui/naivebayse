
from naivebayse import fit , predict 
from numpy import genfromtxt

def main():
    model , classesprob= fit('./weather.csv', target_name='play')
    #print(model)

    ins = genfromtxt( './weather.csv', delimiter=',', dtype = str )
    total = 0 
    count = 0 
    result = []
    orilabel = []
    for i in ins[1:]:
        testing = i[0:len(i) - 1 ]
        prediction = predict(model , targetprob = classesprob , testing_instance= testing)
        result.append(prediction)
        orilabel.append(i[len(i)-1])
        count += 1 
        if(prediction == i[len(i)-1]):
            total += 1

    accuracy = float(total)/ float(count)
    print('Prediction',result)
    print('True label', orilabel)
    print('Accuracy ', accuracy)
  
  
if __name__ == '__main__':
    main()
