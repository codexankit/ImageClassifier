import argparse
import data_utility
import model_utility
from data_utility import load_data

parser = argparse.ArgumentParser(description='Training a model')
parser.add_argument('data_directory', help='Path to dataset on which the model has to be trained')
parser.add_argument('--save_dir', help='Path to directory where the checkpoint should be saved')
parser.add_argument('--arch', help='Network architecture (available option: densenet121, densenet161, densenet169, densenet201, vgg16, vgg19, alexnet (if no architecture is selected then by default vgg16 will be chosen))')
parser.add_argument('--learning_rate', help='Learning rate')
parser.add_argument('--hidden_units', help='Number of neurons in the hidden units')
parser.add_argument('--epochs', help='Number of epochs')
parser.add_argument('--gpu', help='Use GPU for training', action='store_true')


args = parser.parse_args()

# giving deafult values if no input is given
save_dir = '' if args.save_dir is None else args.save_dir
network_architecture = 'vgg16' if args.arch is None else args.arch
learning_rate = 0.0025 if args.learning_rate is None else int(args.learning_rate)
hidden_units = 512 if args.hidden_units is None else float(args.hidden_units)
epochs = 5 if args.epochs is None else int(args.epochs)
gpu = False if args.gpu is None else True


train_data, trainloader, validloader, testloader = load_data(args.data_directory)


model = model_utility.build_network(network_architecture, hidden_units)
model.class_to_idx = train_data.class_to_idx

model, criterion = model_utility.train_network(model, epochs, learning_rate, trainloader, validloader, gpu)
model_utility.evaluate_model(model, testloader, criterion, gpu)
model_utility.save_model(model, network_architecture, hidden_units, epochs, learning_rate, save_dir)