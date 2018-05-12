import pandas as pd
import numpy as np

def cal_classes(data):
    classes = []
    classescount = []
    for att in data:
        if att not in classes:
            classes.append(att)

    for class_value in classes:
        counter = 0
        for att in data:
            if (class_value == att):
                counter = counter + 1
        classescount.append(counter)
    return classes, classescount 

def classes_prob(data):
    classes, classescount = cal_classes(data)
    total = 0
    for i in range(len(classescount)):
        total = total + classescount[i]

    prob = []
    for i in range(len(classescount)):
        probability = float(classescount[i]) / float(total)
        prob.append([classes[i],round(probability, 2)])
    return prob # probability of each class value 

def att_prob(data=[], target=[]):
    if len(data) == len(target):
        classes, classescount = cal_classes(target)
        result = []
        for idx, att in enumerate(target):
            for i, x in enumerate(classes):
                count = 0
                if (att == x):
                    count += 1

                count = float(count) / float(classescount[i])
                count = round(count, 2)
                result.append([data[idx], x, count])
        tempatt = []
        for x in result:
            if x[0] not in tempatt:
                tempatt.append(x[0])
        table = []
       
        for x in tempatt:
            for classname in classes:
                prob = 0
                for y in result:
                    if (y[0] == x and y[1] == classname):
                        prob += y[2]
                table.append([x,classname, round(prob,4)] )
        
        return table, classes, classescount # probability of each attributes value given by each class
       
    else:
        return ('Data', len(data), 'not equal', 'target ', len(target))

def get_probtable(data=[], target=[]):
    att_table, classes, classescount = att_prob(data, target)
    table = []
    for x in classes:
        value = []
        for prob in att_table:
            if (prob[1] == x):
                value.append(prob)
        table.append(value)
  
    return table , classes, classescount

def fit(csv_path , target_name='class'):
    prob_tables = []
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.lower()
    target_name = target_name.lower()
    newdata = data.drop(labels= target_name , axis=1)
    classprob = classes_prob(data[target_name])
    for attributes in newdata:
        table = []
        table , classes, classescount = get_probtable(newdata[attributes], data[target_name])
        prob_tables.append([attributes,table])
    return prob_tables , classprob

def naive_bayse(data=[], target=[]):
    target_prob = classes_prob(target)
    probabilities, classes, classescount = get_probtable(data, target)
    result = []
    for i, att in enumerate(probabilities):
        value = []
        prob = 1
        for x in att:
            if(x[2] == 0):
                x[2] = 1
            prob = float(prob) * float(x[2])
        prob = float(prob) * float(target_prob[i])
        value = [classes[i], prob]
        result.append(value)

    end_result = []
    for i, x in enumerate(result):
        prob = float(target_prob[i]) * float(x[1])
        end_result.append([x[0], round(prob,4)])
    return end_result

def predict(model=[] ,targetprob=[] , testing_instance = ['r','h','h','f']):
    if(len(model) == len(testing_instance)):
        result= []
        for i in range(len(targetprob)):  
            for x in model:
                prob = x[1][i]
                for y in prob:
                    label= y[1]
                    for ins in testing_instance : 
                        if(y[0] == ins):
                            result.append([label,y[2]])
                            break
        givenprob = []
        for label in targetprob:
            total = 1
            for prob in result:
                if (label[0] == prob[0]):
                    total = total * prob[1]
            givenprob.append([label[0],total])
        
        finalresultvalue = []
        finalresultlabel = []
        for label in targetprob:
            total = 1
            for prob in givenprob: 
                if(label[0] ==  prob[0]):
                    total = label[1] * prob[1]
            finalresultvalue.append(round(total,4))
            finalresultlabel.append(label[0])
        
        prediciton = np.argmax([finalresultvalue])
        prediciton = [finalresultlabel[prediciton] , finalresultvalue[prediciton]]
        return prediciton
    else:
        return ('Input lenght: ' , len(testing_instance), 'not equal to ', len(model))
      


