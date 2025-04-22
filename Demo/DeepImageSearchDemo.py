### Sample Dataset Link:
### https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

from DeepImageSearch import Load_Data, Search_Setup

# Reference image
ref_image_path = 'image_path_1'

# Load images from a folder
image_list = Load_Data().from_folder(['folder_path'])

# Set up the search engine
st = Search_Setup(image_list=image_list, model_name='vgg19', pretrained=True, image_count=100)

# Index the images
st.run_index()

# Get metadata
metadata = st.get_image_metadata_file()

# Add New images to the index
st.add_images_to_index(['image_path_1', 'image_path_2'])

# Get similar images
results = st.get_similar_images(image_path=ref_image_path, number_of_images=10)
print(f"Reference image: {ref_image_path}:")
for match in results:
    print(f"  Similar image: {match['image_path']}, Similarity score: {match['score']}")

# Plot similar images
st.plot_similar_images(image_path=ref_image_path, number_of_images=9)

# Update metadata
metadata = st.get_image_metadata_file()