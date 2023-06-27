import time
import numpy as np
import tflite_runtime.interpreter as tflite
import argparse
import pickle

def evaluate_model_tf(model_path, x_test, y_test):
    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    inference_time = []
    for test_sample in x_test:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_sample = np.expand_dims(test_sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_sample)

        # Run inference and time it.
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()

        inference_time.append(end_time - start_time)

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == y_test[index]:
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_digits)

    # Calculate average inference time
    avg_inference_time = np.mean(inference_time)

    return accuracy, avg_inference_time

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
    accuracy, avg_inference_time = evaluate_model_tf(args.model_path, x_test, y_test)

    print(f"Model accuracy: {accuracy}")
    print(f"Average inference time: {avg_inference_time}")
