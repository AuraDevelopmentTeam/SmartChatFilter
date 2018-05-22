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

import dev.aura.smartchatfilter.nn.rating.MessageRating;
import java.io.File;
import java.io.IOException;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class Network {
  private static final int hiddenLayerCount = 10_000;
  private static final int backDropLength = 100;

  private final MultiLayerNetwork network;

  public static Network generateNew() {
    return new Network();
  }

  public static Network loadFromFile(File saveFile) throws IOException {
    return new Network(saveFile);
  }

  private Network() {
    network = new MultiLayerNetwork(getConfiguration());
    network.init();
  }

  private Network(File saveFile) throws IOException {
    network = ModelSerializer.restoreMultiLayerNetwork(saveFile);
  }

  public void saveToFile(File saveFile) throws IOException {
    ModelSerializer.writeModel(network, saveFile, true);
  }

  public void evaluateString(String text) {
    // TODO
  }

  private MultiLayerConfiguration getConfiguration() {
    // TODO: Get this configuration right!
    return new NeuralNetConfiguration.Builder()
        .seed(12345)
        .l2(0.001)
        .weightInit(WeightInit.XAVIER)
        .updater(new RmsProp(0.1))
        .list()
        .layer(
            0,
            new GravesLSTM.Builder()
                .nIn(StringIterator.CHARACTER_COUNT)
                .nOut(hiddenLayerCount)
                .activation(Activation.TANH)
                .build())
        .layer(
            1,
            new GravesLSTM.Builder()
                .nIn(hiddenLayerCount)
                .nOut(hiddenLayerCount)
                .activation(Activation.TANH)
                .build())
        //MCXENT + softmax for classification
        .layer(
            2,
            new RnnOutputLayer.Builder(LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(hiddenLayerCount)
                .nOut(MessageRating.SCORES_COUNT)
                .build())
        .backpropType(BackpropType.TruncatedBPTT)
        .tBPTTForwardLength(backDropLength)
        .tBPTTBackwardLength(backDropLength)
        .pretrain(false)
        .backprop(true)
        .build();
  }
}
