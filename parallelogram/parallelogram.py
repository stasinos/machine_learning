import pandas
import random
import math
import pydot
import statistics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.tree import export_graphviz

print_details=False
export_tree = False
n_folds = 20

df_cols = ['x1','x2','y1','y2','abs_w','abs_h','rel_w','rel_h','max_dim','min_dim','label']

featureset = 1
if featureset == 0:
    # all features
    features = ['x1','x2','y1','y2','abs_w','abs_h','rel_w','rel_h','max_dim','min_dim']
elif  featureset == 1:
    # no position features, position in the grid should not matter
    features = ['abs_w','abs_h','rel_w','rel_h','max_dim','min_dim']
elif  featureset == 2:
    # really easy task, only the correct features given
    features = ['rel_w','rel_h']


def get_label( o ):
    (x1,y1,x2,y2) = o
    if x2 <= x1: raise ValueError
    if y2 <= y1: raise ValueError
    if x2 - x1 == y2 - y1: return 'square'
    else: return 'long'

def extract_features( o ):
    (x1,y1,x2,y2) = o
    abs_w = x2 - x1
    abs_h = y2 - y1
    rel_w = math.ceil( abs_w/abs_h )
    rel_h = math.ceil( abs_h/abs_w )
    max_dim = max( [abs_w,abs_h] )
    min_dim = min( [abs_w,abs_h] )
    return [abs_w,abs_h,rel_w,rel_h,max_dim,min_dim]

def make_object( x1, y1, max_dim=100 ):
    if max_dim<2: raise ValueError
    lim_x2 = min( [x1+max_dim,99] )
    x2 = random.randint( x1+1, lim_x2 )
    lim_y2 = min( [y1+max_dim,99] )
    y2 = random.randint( y1+1, lim_y2 )
    return ( x1, y1, x2, y2 )

def make_square( x1, y1, max_dim=100 ):
    if max_dim<2: raise ValueError
    lim = min( [max(x1,y1)+max_dim, 99] )
    delta = random.randint( max(x1,y1)+1, lim )
    x2 = x1+delta
    y2 = y1+delta
    return ( x1, y1, x2, y2 )

def make_long( x1, y1, max_dim=100 ):
    if max_dim<2: raise ValueError
    # x1,y1 at the edge, impossible to fit long
    if x1 == 98 and y1 == 98: raise ValueError
    lim_x2 = min( [x1+max_dim,99] )
    lim_y2 = min( [y1+max_dim,99] )
    delta_x = random.randint( x1+1, lim_x2 )
    delta_y = random.randint( y1+1, lim_y2 )
    while delta_x == delta_y:
        delta_x = random.randint( x1+1, lim_x2 )
        delta_y = random.randint( y1+1, lim_y2 )
    x2 = x1+delta_x
    y2 = y1+delta_y
    return ( x1, y1, x2, y2 )


def make_training_dataset( n ):
    data = []
    n_sq = 0
    for i in range(0,n):
        x1 = random.randint( 0, 98 )
        y1 = random.randint( 0, 98 )
        o = make_object( x1, y1, 12 )
        f = extract_features( o )
        (x1,y1,x2,y2) = o
        datapoint = [ x1, y1, x2, y2 ]
        datapoint.extend( f )
        l = get_label( o )
        if l == 'square': n_sq += 1
        datapoint.append( l )
        data.append( datapoint )
    if print_details: print( "Made dataset with {:d}/{:d} squares".format(n_sq,n) )
    return pandas.DataFrame( data, columns=df_cols )


def make_production_dataset( n ):
    data = []
    for i in range(0,math.ceil(n/2)):
        x1 = random.randint( 0, 97 )
        y1 = random.randint( 0, 97 )
        try:
            o = make_long( x1, y1, 72 )
        except:
            # x1,y1 at the edge,
            # impossible to fit long
            x1 = random.randint( 0, 90 )
            y1 = random.randint( 0, 90 )
            o = make_long( x1, y1 )
        data.append( o )
    for i in range(0,math.ceil(n/2)):
        x1 = random.randint( 0, 98 )
        y1 = random.randint( 0, 98 )
        o = make_square( x1, y1, 72 )
        data.append( o )
    dataset = []
    n_sq = 0
    for o in data:
        (x1,y1,x2,y2) = o
        datapoint = [ x1, y1, x2, y2 ]
        f = extract_features( o )
        datapoint.extend( f )
        l = get_label( o )
        if l == 'square': n_sq += 1
        datapoint.append( l )
        dataset.append( datapoint )
    return pandas.DataFrame( dataset, columns=df_cols )


def make_one_experiment( n ):
    if n <20: raise ValueError
    df = make_training_dataset( n )
    X = df[features].values
    y = df.label.values

    # Split into n_folds folds and use n_folds-1 for training
    # and 1 for validation. Return the best model among these
    # n_folds models as the final outcome.
    kf = KFold( n_splits=n_folds, shuffle=False, random_state=None )
    tests1 = []
    tests2 = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = DecisionTreeClassifier( max_depth=6 )
        # For unlimited tree depth:
        #clf = DecisionTreeClassifier()
        clf = clf.fit( X_train, y_train )
        y_pred = clf.predict( X_train )
        acc_train = metrics.accuracy_score(y_train, y_pred)
        y_pred = clf.predict( X_test )
        acc_test = metrics.accuracy_score(y_test, y_pred)
        tests1.append( acc_train )
        tests2.append( acc_test )
    # Folding is only used to estimate the accuracy
    # Now use all the data to train the classifier
    # that will be deployed.
    avg_acc_train = statistics.mean( tests1 )
    avg_acc_test = statistics.mean( tests2 )
    clf = DecisionTreeClassifier( max_depth=6 )
    clf = clf.fit( X, y )

    df = make_production_dataset( 1000 )
    X = df[features].values
    y = df.label.values
    pred = clf.predict( X )
    actual_acc = metrics.accuracy_score( y, pred )
    difference = actual_acc - avg_acc_test

    if print_details:
        print( "Accuracy: from {:.2f} (training) to {:.2f} (validation) to {:.2f} (actual) -> {:.2f} (difference)".format(avg_acc_train, avg_acc_test,
                actual_acc, difference) )
        
        if export_tree:
            # This creates a tree.dot file that can then be
            # used to create an image of the decision tree
            export_graphviz(clf, "tree.dot", filled=True, rounded=True,
                    special_characters=True, feature_names=features, class_names=['long','square'])
            graphs = pydot.graph_from_dot_file( "tree.dot" )
            graphs[0].write_png( "tree.png" )

    return difference

random.seed( 12 )

for n in [100,500,1000,2500,5000]:
    count = 0
    for i in range(0,100):
        if print_details: print("Experiment " + str(i) )
        acc_diff = make_one_experiment( n )
        if acc_diff < -0.2: count += 1
    print("With {:d} training examples, accuracy drop > 20% happens at {:d}% of the runs".format(n,count) )
