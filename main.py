
from naivebayse import fit , predict 

def main():
    model , classesprob= fit('./weather.csv', target_name='play')
    #print(model)
    prediction = predict(model , targetprob = classesprob , testing_instance=['o','h','h','f'])
    print(prediction)
  
  
if __name__ == '__main__':
    main()
