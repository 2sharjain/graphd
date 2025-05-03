import tensorflow as tf

# Specify the path to your .h5 file
model_path = '/media/raghavendra/12f6e4a1-dfd9-43d5-b35a-f46c4f2e139b/raghavendra/GNN/DNN2/DNN2_implementation_example/models_h5/DNN2_implementation_example_04.h5'

# Load the model
model = tf.keras.Model.load_model(model_path)

# Now you can use the model to make predictions or examine its structure
# Example of predicting with some input data
# predictions = model.predict(your_input_data)

# Print the summary of the model architecture
print(model)