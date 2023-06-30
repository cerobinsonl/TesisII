import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
import argparse
import pickle

def evaluate_model_tf(model_path, x_test, y_test):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    inference_time = []
    for test_sample in x_test:
        test_sample = np.expand_dims(test_sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_sample)

        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()

        inference_time.append(end_time - start_time)

        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)
    
    # Convert your test and prediction data to numpy arrays if they are not already
    y_test_np = np.array(y_test)
    prediction_digits_np = np.array(prediction_digits)

    # Compute precision, recall, and F1 score
    precision = precision_score(y_test, 
                                prediction_digits_np, 
                                average="weighted")
    
    # Calculate average inference time
    avg_inference_time = np.mean(inference_time)

    # Compute confusion matrix
    conf_mat = confusion_matrix(y_test_np, prediction_digits_np)

    # Compute F1-score
    f1 = f1_score(y_test_np, prediction_digits_np, average='weighted')

    # Compute recall
    recall = recall_score(y_test_np, prediction_digits_np, average='weighted')

    return precision, avg_inference_time, conf_mat, f1, recall

def load_data(x_path, y_path):
    with open(x_path, 'rb') as f:
        x_test = pickle.load(f)
    with open(y_path, 'rb') as f:
        y_test = pickle.load(f)
    return x_test, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a TFLite model.')
    parser.add_argument('model_path', type=str, help='The path to the TFLite model file.')
    parser.add_argument('x_test_path', type=str, help='The path to the pickle file containing the x_test data.')
    parser.add_argument('y_test_path', type=str, help='The path to the pickle file containing the y_test data.')
    args = parser.parse_args()

    x_test, y_test = load_data(args.x_test_path, args.y_test_path)
    precision, avg_inference_time, conf_mat, f1, recall = evaluate_model_tf(args.model_path, x_test, y_test)

    print(f"Model precision: {precision}")
    print(f"Average inference time: {avg_inference_time}")
    print(f"Confusion Matrix: \n{conf_mat}")
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
