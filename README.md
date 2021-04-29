# Scene-Classification-with-BoW-and-kNN

## Introduction and Overview

One of the core problems in computer vision is classification.  Given an image that comes from a few fixed categories, can you determine which category it belongs to?  For this assignment, you will be developing a system for scene classification.  A system like this might be used by services that have a lot of user uploaded photos, like Flickr or Instagram, that wish to automatically categorize photos based on the type of scenery.  It could also be used by a robotic system to determine what type of environment it is in, and perhaps change how it moves around.

You have been given a subset of the SUN Image database consisting of eight scene categories.  See Figure 1.  In this assignment, you will build an end to end system that will, given a new scene image, determine which type of scene it is.

This assignment is based on an approach to document classification called Bag of Words.  This approach to document classification represents a document as a vector or histogram of counts for each word that occurs in the document, as shown in Figure 2.  The hope is that different documents in the same class will have a similar collection and distribution of words, and that when we see a new document, we can find out which class it belongs to by comparing it to the histograms already in that class.  This approach has been very successful in Natural Language Processing, which is surprising due to its relative simplicity.  We will be taking the same approach to image classification.  However, one major problem arises.  With text documents, we actually have a dictionary of words.  But what words can we use to represent an image with?


This assignment has 3 major parts.  Part 1 involves building a dictionary of visual words from the training data.  Part 2 involves building the recognition system using our visual word dictionary and our training images.  In Part 3, you will evaluate the recognition system using the test images.

In Part 1, you will use the provided filter bank to to convert each pixel of each image into a high dimensional representation that will capture meaningful information, such as corners, edges etc...  This will take each pixel from being a 3D vector of color values, to an nD vector of filter responses.  You will then take these nD pixels from all of the training images and and run K-means clustering to find groups of pixels.  Each resulting cluster center will become a visual word, and the whole set of cluster centers becomes our dictionary of visual words.  In theory, we would like to use all pixels from all training images, but this is very computationally expensive.  Instead, we will only take a small sample of pixels from each image.  One option is to simply select α pixels from each one uniformly at random.  Another option is to use some feature detector (Harris Corners for example), and take α feature points from each image.  You will do both to produce two dictionaries, so that we can compare their relative performances.  See Figure 3.

In Part 2, the dictionary of visual word you produced will be applied to each of the training images to convert them into a word map.  This will take each of the nD pixels in all of the filtered training images and assign each one a single integer label, corresponding to the closest cluster center in the visual words dictionary.  Then each image will be converted to a "bag of words";  a histogram of the visual words counts.  You will then use these to build the classifier.  See Figure 4.

In Part 3, you will evaluate the recognition system that you built. This will involve taking the test images and converting them to image histograms using the visual words dictionary and the function you wrote in Part 2. Next, for nearest neighbor classification, you will use a histogram distance function to compare the new test image histogram to the training image histograms in order to classify the new test image. Doing this for all the test images will give you an idea of how good your recognition system. See Figure 5.



## Programming Problems

The necessary code files for getting started are provided for you, as well as the pertinent sub-set of the SUN dataset.  It can all be downloaded in this zip file:  scene_classification.zip download

### Part 1: Build Visual Words Dictionary

#### Q1.1  Extract Filter Responses

In the file extractFilterResponses.py a function to extract filter responses:
filterResponses = extract_filter_responses (img, filterBank)

We have provided the function createFilterBank().  This function will generate a set of image convolution filters. See Figure 6.  There are 4 filters, and each one is made at 5 different scales, for a total of 20 filters.  The filters are 4:

- Gaussian:  Responds strongly to constant regions, and suppresses edges, corners and noise
- Laplacian of Gaussian:  Responds strongly to blobs of similar size to the filter
- X Gradient of Gaussian:  Responds strongly to vertical edges
- Y Gradient of Gaussian:  Responds strongly to horizontal edges


Pass this array of image convolution filters to your extract_filter_responses(..) function, along with an image of size H x W x 3.  Your function should convert the color space of img from BGR to Lab.  Then it should apply all of the n filters on each of the 3 color channels of the input image.  You should end up with 3n filter responses for the image.  For image filtering, you can use the provided helper function cv2.filter2D(..).  The final matrix that you return should have size H x W x 3n  (Figure 7).

With your submission, include an image from the data set and 3 of its filter responses.

#### Q1.2 Collect sample of points from image

Write two functions that return a list of points in an image, that will then be used to generate visual words.  These functions will be used in the next part to select points from every training image.

First, write a simple function that takes an image and an α, and returns a matrix of size α x 2 of random pixels locations inside the image (i.e. α entires with 2 values per pixel:  x and y) 

Write two functions that return a list of points in an image, that will then be used to generate visual words.  These functions will be used in the next part to select points from every training image.

First, write a simple function that takes an image and an α, and returns a matrix of size α x 2 of random pixels locations inside the image (i.e. α entires with 2 values per pixel:  x and y) 

points = get_random_points (img, alpha)

Next, write a function that uses the Harris corner detection algorithm to select key points from an input image:

points = get_harris_points (img, alpha, k)

Feel free to use cv2.cornerHarris(..), but remember that it returns an array the same size as the input image that indicates the corner strength at each pixel.  Your function should return a list of the top α points with the highest corner strength (e.g. if alpha=50, return the 50 points with the top strength).

You can use 3 x 3 or 5 x 5 for the Sobel aperture size, and a good value for the k parameter is 0.04 - 0.06.

#### Q1.3  Compute Dictionary of Visual Words


You will now create the dictionary of visual words.  Write the function:

dictionary = get_dictionary (imgPaths, alpha, K, method)

This function takes in an array of training image paths.  For each and every training image, load it and apply the filter bank to it.  Then get α points for each image and put them into an array, where each row represents a single n-dimensional pixel (n is the number of filters).  If there are T training images total, then you will build a matrix of size αT x 3n, where each row corresponds to a single pixel, and there are αT total pixels collected.

method will be a string either 'Random' or 'Harris', and will determine how the α points will be taken from each image.  If the method is 'Random', then the α points will be selected using get_random_points(..), while method 'Harris' will tell your function to use get_harris_points(..) to select the α points.  Once you have done this, pass pixel responses of all the points collected from all training images to sklearn's K-means (Links to an external site.) clustering algorithm (already implemented), which finds clusters that correspond to visual words in the dictionary.

The result will be a matrix of size K x 3n, where each row represents the coordinates of a cluster center.  This matrix will be your dictionary of visual words.  For testing purposes, use α = 50 and K = 100.  Eventually you will want to use much larger numbers (e.g., α = 200 and K = 500) to get better results for the final part of the write-up.

This function can take a while to run.  Start early and be patient.  When it has completed, you will have your dictionary of visual words. Save this in a pickle (.pkl)  file.  This is your visual words dictionary.  It may be helpful to write a computeDictionary.py, which will do the legwork of loading the training image paths, processing them, building the dictionary, and saving the pickle file. This is not required, but will be helpful to you when calling it multiple times.

For this question, you must produce two dictionaries.  One named dictionaryRandom.pkl, which used the random method to select points and another named dictionaryHarris.pkl which used the Harris method.  Both must be handed in.


### Part 2: Build Visual Scene Recognition System

#### Q2.1  Convert image to word map

Write a function to map each pixel in the image to its closest word in the dictionary.

wordMap = get_visual_words (img, filterBank, dictionary)

img is the input image of size H x W x 3.  dictionary is the dictionary computed previously.  filterBank is the filter bank that was used to construct the dictionary.  wordMap is an H x W matrix of integer labels, where each label corresponds to a word/cluster center in the dictionary.

Use scipy's function cdist(..) with Euclidean distance to do this efficiently.  You can visualize your results with the skimage.color function label2rgb(..) (then convert to BGR for cv2.imshow()).

Once you are done, call the provided script batchToVisualWords.py.  This function will apply your implementation of get_visual_words(..) to every image in the training and testing set.  The script will load traintest.pkl, which contains the names and labels of all the data images.  For each training image data/<category>/X.jpg, this script will save the word map pickle file data/<category>/X <point method>.pkl.  For optimal speed, modify num_cores to the number of cores in your computer.  This will save you from having to keep re-running get_visual_words(..) in the later parts, unless you decide to change the dictionary.

With your submission, turn in the word maps for 3 different images from two different classes.  Do this for each of the two dictionary types (random and Harris) -- so there will be 12 images in total.

#### Q2.2  Get Image Features

Create a function that extracts the histogram of visual words within the given image (i.e., the bag of visual words)

h = get_image_features (wordMap, dictionarySize)

h, the vector representation of the image, is a histogram of size 1 x K, where dictionarySize is the number of clusters K in the dictionary.  h(i) should be equal to the number of times visual word i occurred in the word map.  Since the images are of differing sizes, the total count in h will vary from image to image.  To account for this, L1 normalize the histogram, relative to the size of the image, before returning it from your function.

#### Q2.3  Build Recognition System - Nearest Neighbors


Now that you have built a way to get a vector representation for an input image, you are ready to build the visual scene classification system.  This classifier will make use of nearest neighbor classification.  Write a script buildRecognitionSystem.py that saves visionRandom.pkl and visionHarris.pkl and in each pickle store a dictionary that contains:

dictionary:  your visual word dictionary, a matrix of size K x 3n
filterBank:  filter bank used to produce the dictionary.  This is an array of image filters
trainFeatures:  T x K matrix containing all of the histograms of visual words of the T training images in the data set.
trainLabels:  T x 1 vector containing the labels of each training image.
You will need to load the train_imagenames and train_labels from traintest.pkl.  Load dictionary from dictionaryRandom.pkl and dictionaryHarris.pkl you saved in part Q1.3.



### Part 3: Evaluate Visual Scene Recognition System

#### Q3.1 Image Feature Distance

For nearest neighbor classification you need a function to compute the distance between two image feature vectors. Write a function

dist = get_image_distance (hist1, hist2, method)

hist1 and hist2 are the two image histograms whose distance will be computed (with hist2 being the target), and returned in dist.  The idea is that two images that are very similar should have a very small distance, while dissimilar images should have a larger distance. method will control how the distance is computed, and will either be set to 'euclidean' or 'chi2'.  The first option tells to compute the Euclidean distance between the two histograms.  The second uses χ2 distance. For χ2 distance, you can use the function chi2dist(..) in utils.py.   Alternatively, you may also write the function

[dist] = get_image_distance (hist1, histSet, method)

which, instead of the second histogram, takes in a matrix of histograms, and returns a vector of distances between hist1 and each histogram in histSet.  This may make it possible to implement things more efficiently.  Of course, you need to modify chi2dist(..) to handle the calculation of multiple distances.  Choose either one of the two get_image_distance(..) options to hand in.

#### Q3.2  Evaluate Recognition System - NN and kNN

Write a script evaluateRecognitionSystem_NN.py that evaluates your nearest neighbor recognition system on the test images.  Nearest neighbor classification assigns the test image the same class as the "nearest" sample in your training set.  "Nearest" is defined by your distance function.

Load traintest.pkl and classify each of the test_imagenames  files.  Have the script report both the accuracy , as well as the 8 x 8 confusion matrix C, where the entry C(i,j) records the number of times an image of actual class i was classified as class j.

The confusion matrix can help you identify which classes work better than others and quantify your results on a per-class basis.  In a perfect classifier, you would only have entries in the diagonal of C (implying, for example, that an 'auditorium' always got correctly classified as an 'auditorium').  For each combination of dictionary (random or Harris) and distance metric (Euclidean and χ2), have your script print out the confusion metric and the confusion matrix.  Use print(..) so we can know which is which.

Now take the best combination of dictionary and distance metric, and write a script evaluateRecognitionSystem_kNN.py that classifies all the test images using k Nearest Neighbors. Have your script generate a plot of the accuracy for k from 1 to 40. For the best performing k value, print the confusion matrix. Note that this K is different from the dictionary size k.  This k is the number of nearby points to consider when classifying the new test image.
