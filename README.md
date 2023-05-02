# Self_Driving_Car
we are going to know how to train a self driving car using Convolution neural networks CNN. We will be using the open source Self driving car simulator provided by Udacity that is used in their Self driving car Nano degree program. Using this simulator we will first drive the car and collect data. Then we will train a CNN model to learn this behavior and then test it back on the simulator. The model we will use was proposed by NVIDIA. They used this model to train a real car data and got promising results when they drove it autonomously.


# Udacity-self-driving-car
This is a machine learning project, in which a car is driven autonomously in a simulator using a nine-layered convolutional neural network. The simulator used in the video is Udacity's open source simulator. Find the project video on [YouTube](https://www.youtube.com/watch?v=I39Zn5ip_nQ)

# The Concept
So how do humans learn to drive. Do we calculate how much to turn based on the road lane? Does it have to be specific color road lanes? what if there are no road lanes? Well we will still be able to drive. So the question is how did we learn? For years and years we watch and observe i.e collecting data of parents driving going to road trips going to school learning the basics of driving. And once we finally get behind the we have a basic understanding of roads, lanes different signs etc. So at this point we already have a model in our head that knows the basics but once we start driving ourselves we train the model even further this time will steering information along with acceleration and braking. So withing a few days of training we learn how to drive. So over time we keep getting better as we keep getting more data. So the key here is that we collect data and based on this we create a model that generalizes how to drive.

Below we can see the System Structure of NVIDIA’s method, where they collected 3 images form different cameras and recorded the steering angle for these images.

![structure](https://user-images.githubusercontent.com/119833997/235556149-cacd091d-d28a-47da-aa58-ae455a26c68c.png)

## libraries
* [Udacity open source simulator](https://github.com/udacity/self-driving-car-sim)
* [Tensorflow-gpu](https://www.tensorflow.org/install/) - Deep learning library used.
* [TFlearn](https://tflearn.org/) - Higher level wrapper on tensorflow.

### Installing
* clone this repository.
* [Tensorflow-gpu](https://www.tensorflow.org/install/) - Install tensorflow gpu from this link (you will need a nvidia graphics card)
* install [OpenCV](https://opencv.org/)
* install [Numpy](https://numpy.org/)
* install [Matplotlib](https://matplotlib.org/)
* useing the following command to install other dependencies.
```
pip install -r requirements.txt
```




## Checking the installation
To check the installation of the libraries you can import them in your python terminal. 


## How to train your own neural network
create IMG, data folder
save my images in IMG folder and save the driving_log.csv in data folder then run the following command.

```
python model_train.py
```

## Testing

Start the simulator in autonomous mode. Then run the following command.
```
python drive.py
```
## Authors

* **Assim Abdelnour** - [github](https://github.com/Assim22)

* **Atef Selim** - [github](https://github.com/Atefselim)
* **Salah Ahmed**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
