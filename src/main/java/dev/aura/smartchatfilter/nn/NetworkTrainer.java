/*
 * SmartChatFilter - A very WIP plugin that rates and filters chat based on a neural network.
 * Copyright (C) 2018  Yannick Schinko
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package dev.aura.smartchatfilter.nn;

import dev.aura.smartchatfilter.log.AdvancedScoreIterationListener;
import java.io.IOException;
import lombok.Getter;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

@Log4j2
public class NetworkTrainer {
  @Getter private boolean running;
  private Thread thread;

  public void start() {
    if (running) {
      return;
    }

    logger.info("Starting NetworkTrainer...");
    thread = new Thread(this::runTraining);
    thread.start();

    running = true;
    logger.debug("Started NetworkTrainer!");
  }

  public void stop() {
    if (!running) {
      return;
    }

    logger.debug("Stopping NetworkTrainer...");
    running = false;
    thread.interrupt();

    logger.info("Stopped NetworkTrainer!");
  }

  @SneakyThrows(IOException.class)
  public void runTraining() {
    //number of rows and columns in the input pictures
    final int numRows = 28;
    final int numColumns = 28;
    int outputNum = 10; // number of output classes
    int batchSize = 128; // batch size for each epoch
    int rngSeed = 123; // random number seed for reproducibility
    int numEpochs = 100; // number of epochs to perform

    //Get the DataSetIterators:
    DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
    DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

    logger.info("Build model....");
    MultiLayerConfiguration conf =
        new NeuralNetConfiguration.Builder()
            .seed(rngSeed) //include a random seed for reproducibility
            // use stochastic gradient descent as an optimization algorithm
            .updater(new Nesterovs(0.006, 0.9))
            .l2(1e-4)
            .list()
            .layer(
                0,
                new DenseLayer.Builder() //create the first, input layer with xavier initialization
                    .nIn(numRows * numColumns)
                    .nOut(1000)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .layer(
                1,
                new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                    .nIn(1000)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .pretrain(false)
            .backprop(true) //use backpropagation to adjust weights
            .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    //print the score with every 100 iteration (and every 10 to the log)
    model.setListeners(new AdvancedScoreIterationListener(100, 10));

    logger.info("Train model....");
    for (int i = 0; i < numEpochs; i++) {
      model.fit(mnistTrain);
    }

    logger.info("Evaluate model....");
    Evaluation eval =
        new Evaluation(outputNum); //create an evaluation object with 10 possible classes
    while (mnistTest.hasNext()) {
      DataSet next = mnistTest.next();
      INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
      eval.eval(next.getLabels(), output); //check the prediction against the true class
    }

    logger.info(eval.stats());
    logger.info("****************Example finished********************");
  }
}
