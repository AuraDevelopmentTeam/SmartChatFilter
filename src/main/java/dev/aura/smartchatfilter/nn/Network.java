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
import dev.aura.smartchatfilter.nn.rating.MessageRating;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.LinkedList;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class Network {
  private static final int hiddenLayerCount = 1_000;
  private static final int backDropLength = 100;

  private final MultiLayerNetwork network;

  public static Network generateNew() {
    return new Network();
  }

  public static Network loadFromFile(File saveFile) throws IOException {
    if (saveFile.exists()) return new Network(saveFile);
    else return generateNew();
  }

  private Network() {
    network = new MultiLayerNetwork(getConfiguration());
    network.init();
    network.setListeners(new AdvancedScoreIterationListener(10, 5));
  }

  private Network(File saveFile) throws IOException {
    network = ModelSerializer.restoreMultiLayerNetwork(saveFile);
  }

  public void saveToFile(File saveFile) throws IOException {
    final File parent = saveFile.getParentFile();

    if (!parent.exists() && !parent.mkdirs())
      throw new IOException("Could not create directory: " + parent);

    ModelSerializer.writeModel(network, saveFile, true);
  }

  public MessageRating evaluateString(String text) {
    final INDArray output = network.output(new StringIterator(text, new MessageRating(0, 0, 0)));
    final INDArray dataOutput =
        output.getRow(0).getColumn(text.getBytes(StandardCharsets.UTF_8).length - 1);

    return new MessageRating(
        dataOutput.getDouble(0), dataOutput.getDouble(1), dataOutput.getDouble(2));
  }

  public void train(DataSetIterator iterator) {
    network.fit(iterator);
  }

  public Evaluation evaluate(DataSetIterator iterator) {
    return network.evaluate(iterator);
  }

  public void enableUI() {
    final UIServer uiServer = UIServer.getInstance();
    final StatsStorage statsStorage = new InMemoryStatsStorage();

    uiServer.attach(statsStorage);

    // network.addListeners is bugged
    final Collection<TrainingListener> listeners = new LinkedList<>(network.getListeners());
    listeners.add(new StatsListener(statsStorage, 5));
    network.setListeners(listeners);
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
            new LSTM.Builder()
                .nIn(StringIterator.CHARACTER_COUNT)
                .nOut(hiddenLayerCount)
                .activation(Activation.TANH)
                .build())
        .layer(
            1,
            new LSTM.Builder()
                .nIn(hiddenLayerCount)
                .nOut(hiddenLayerCount)
                .activation(Activation.TANH)
                .build())
        .layer(
            2,
            new RnnOutputLayer.Builder(LossFunction.MSE)
                .activation(Activation.SIGMOID)
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
