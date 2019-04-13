from keras.models import load_model,Sequential
from keras.layers import Dense
import argparse

def regressor(model):
    new_model = Sequential(model.layers[:-1])
    for layer in new_model.layers:
        layer.trainable = False
    new_model.add(Dense(1,activation='relu'))
    return new_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--infile',action='store',help='Input file')
    parser.add_argument('-o','--outfile',action ='store',help='Output file')
    args = parser.parse_args()
    
    model = load_model(args.infile)
    new_model = regressor(model)
    new_model.compile(loss='mean_squared_error',optimizer='adam')
    
    data = pd.read_csv(args.infile)
    X = np.array([encode(seq) for seq in data['Consensus sequence']])
    y = data['Kd'].values
    new_model.train(X,y,epochs=100)
    new_model.evaluate(X,y)
    
