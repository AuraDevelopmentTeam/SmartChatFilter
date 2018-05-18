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

import dev.aura.smartchatfilter.Main;
import dev.aura.smartchatfilter.log.AdvancedScoreIterationListener;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import lombok.Getter;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

@Log4j2
public class NetworkTrainer {
  @Getter private boolean running;
  private Thread thread;

  public void start() {
    if (running) {
      return;
    }

    logger.info("Starting NetworkTrainer...");
    thread = new Thread(this::runTraining, getClass().getSimpleName());
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
    final int nChannels = 1; // Number of input channels
    final int outputNum = 10; // The number of possible outcomes
    final int batchSize = 64; // Test batch size
    final int nEpochs = 10; // Number of training epochs
    final int seed = 123; //

    /*
       Create an iterator using the batch size for one iteration
    */
    logger.info("Load data....");
    final DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
    final DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

    /*
       Construct the neural network
    */
    logger.info("Build model....");

    // learning rate schedule in the form of <Iteration #, Learning Rate>
    final Map<Integer, Double> lrSchedule = new HashMap<>();
    lrSchedule.put(0, 0.01);
    lrSchedule.put(1000, 0.005);
    lrSchedule.put(3000, 0.001);
    final Map<Integer, Double> mvSchedule = new HashMap<>();
    mvSchedule.put(0, 0.9);
    mvSchedule.put(1000, 0.8);
    mvSchedule.put(3000, 0.7);

    final MultiLayerConfiguration conf =
        new NeuralNetConfiguration.Builder()
            .seed(seed)
            .l2(0.0005)
            .weightInit(WeightInit.XAVIER)
            .updater(
                new Nesterovs(
                    new MapSchedule(ScheduleType.ITERATION, lrSchedule),
                    new MapSchedule(ScheduleType.ITERATION, mvSchedule)))
            .list()
            .layer(
                0,
                new ConvolutionLayer.Builder(5, 5)
                    //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                    .nIn(nChannels)
                    .stride(1, 1)
                    .nOut(20)
                    .activation(Activation.IDENTITY)
                    .build())
            .layer(
                1,
                new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
            .layer(
                2,
                new ConvolutionLayer.Builder(5, 5)
                    //Note that nIn need not be specified in later layers
                    .stride(1, 1)
                    .nOut(50)
                    .activation(Activation.IDENTITY)
                    .build())
            .layer(
                3,
                new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
            .layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
            .layer(
                5,
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build())
            .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
            .backprop(true)
            .pretrain(false)
            .build();

    /*
    Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
    (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
        and the dense layer
    (b) Does some additional configuration validation
    (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
        layer based on the size of the previous layer (but it won't override values manually set by the user)
    InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
    For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
    MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
    row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
    */

    final MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    logger.info("Train model....");
    //Print score every 10 iterations
    model.setListeners(new AdvancedScoreIterationListener(100, 10));

    for (int i = 0; i < nEpochs; i++) {
      model.fit(mnistTrain);
      logger.info("*** Completed epoch {} ***", i);

      logger.info("Evaluate model....");
      Evaluation eval = model.evaluate(mnistTest);
      logger.info(eval.stats());
      mnistTest.reset();
    }

    logger.info("****************Example finished********************");

    Main.stop();
  }
}
