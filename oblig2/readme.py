"""
Dette er Praktisk Maskinlæring oblig 2!
Forfatter: Stian Larsen
Git: stianlars1


* Describe the datasets

Classification: 
Datasettet jeg har valgt er et utvalg av pasienter som har eller ikke har diabetes, 
dette er da basert på forskjellige egenskaper ved pasientes som:
- Alder
- Bmi
- insulin-nivå
- Vært gravide
- blodtrykk
etc...

Datasettet er strukturert slik at den kun inneholder tallverdier. 
Tallene ser også normaliserte ut, og det finnes ingen rader som inneholder "nullheter" i seg. 



Kernel SVC = linear
Kernel Decision Tree = entropy

Results: 


model 1. 
    SVC model has an Accuracy output of: 0.7835497835497836
    
    Confusion Matrix of SVC model is: 
                                        [[134  19]
                                        [ 31  47]]
        True positives: 134
        False positives: 19
        False negatives: 31
        True negatives: 47
        
    Cross Validation Performance results:
    [0.7037037  0.75925926 0.74766355 0.82242991 0.74766355]
    Cross Validation mean: 0.7561439944617515 accuracy with a standard deviation of 0.038203002339734664
    Train data accuracy:  0.76909
    Test data accuracy:  0.78355
        

model 2. 
    DecisionTreeClassifier model has an Accuracy output of: 0.7272727272727273
    
    Confusion Matrix of DecisionTreeClassifier is:
                                        [[120  33]
                                        [ 30  48]]
        True positives: 120
        False positives: 33
        False negatives: 30
        True negatives: 48
        
    Cross Validation Performance results: 
        [0.68518519 0.65740741 0.6635514  0.76635514 0.6728972 ]
        Cross Validation mean: 0.6890792661820699 accuracy with a standard deviation of 0.03975767209370288
        Train data accuracy:  1.0
        Test data accuracy:  0.72727
        

model 3. 
    GaussianNB model has an Accuracy output of: 0.7748917748917749
    
    Confusion Matrix of DecisionTreeClassifier is:
                                        [[130  23]
                                        [ 29  49]]
        True positives: 130
        False positives: 23
        False negatives: 29
        True negatives: 49
        
    Cross Validation Performance results: 
        [0.68518519 0.73148148 0.71028037 0.80373832 0.80373832]
        Cross Validation mean: 0.7468847352024921 accuracy with a standard deviation of 0.04867983908653887
        Train data accuracy:  0.7635
        Test data accuracy:  0.77489

    
"""




"""
Kernel = rbf
Kernel Decision Tree = gini


model 1. 
    SVC model has an Accuracy output of: 0.7532467532467533

    Confusion Matrix of SVC model is: 
                                        [[138  18]
                                        [ 39  36]]
        True positives: 138
        False positives: 18
        False negatives: 39
        True negatives: 36
    
    Cross Validation Performance results: 
        [0.84259259 0.75       0.71962617 0.75700935 0.74766355]
        Cross Validation mean: 0.7633783316026307 accuracy with a standard deviation of 0.04160684798906006
        Train data accuracy:  0.77281
        Test data accuracy:  0.75325


model 2. 
    DecisionTreeClassifier model has an Accuracy output of: 0.7142857142857143

    Confusion Matrix of DecisionTreeClassifier is:
                                        [[115  41]
                                        [ 25  50]]
        True positives: 115
        False positives: 41
        False negatives: 25
        True negatives: 50
        
    Cross Validation Performance results: 
        [0.74074074 0.64814815 0.64485981 0.75700935 0.71028037]
        Cross Validation mean: 0.7002076843198337 accuracy with a standard deviation of 0.04635601069943756
        Train data accuracy:  1.0
        Test data accuracy:  0.71429


model 3. 
    GaussianNB model has an Accuracy output of: 0.7575757575757576

    Confusion Matrix of DecisionTreeClassifier is:
                                        [[134  22]
                                        [ 34  41]]
        True positives: 134
        False positives: 22
        False negatives: 34
        True negatives: 41
        
    Cross Validation Performance results: 
        [0.78703704 0.74074074 0.73831776 0.81308411 0.70093458]
        Cross Validation mean: 0.7560228452751817 accuracy with a standard deviation of 0.03949581793116758
        Train data accuracy:  0.76164
        Test data accuracy:  0.75758


"""







"""
Comparison results::

Fra hva jeg kan se, og har erfaring gjennom oppgaven som gjelder Classification, 
kan jeg se at modellen som har hatt best accuracy score basert på mange kjøringer er: 


SVC linear:
    SVC model has an Accuracy output of: 0.7835497835497836
    Train data accuracy:  0.76909
    Test data accuracy:  0.78355

                                                                    best accuracy based on kernel = 0.78 with kernel = linear

SVC rbf:
    SVC model has an Accuracy output of: 0.7532467532467533
    Train data accuracy:  0.77281
    Test data accuracy:  0.75325




Decision Tree kernel = Entropy:
    DecisionTreeClassifier model has an Accuracy output of: 0.7272727272727273
    Train data accuracy:  1.0
    Test data accuracy:  0.72727

                                                                    Best accueacy based on criterion = 0.727 with Entropy, low difference tho..

Decision Tree kernel = gini:
    DecisionTreeClassifier model has an Accuracy output of: 0.7142857142857143
    Train data accuracy:  1.0
    Test data accuracy:  0.71429




Gaussian Naive Bayes first run: 
    GaussianNB model has an Accuracy output of: 0.7748917748917749
    Train data accuracy:  0.7635
    Test data accuracy:  0.77489

                                                                    Best accuracy on Naive bayes Gaussian = 0.77 first run. 

Gaussian Naive Bayes second run: 
    GaussianNB model has an Accuracy output of: 0.7575757575757576
    Train data accuracy:  0.76164
    Test data accuracy:  0.75758




Results show that the best accuracy for this dataset is based on the SVC "Supported Vector Machines algorith", with an accuracy on 78.3% chance on detecting diabetes on pasients. 

"""