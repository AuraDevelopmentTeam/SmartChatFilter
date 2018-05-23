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
import dev.aura.smartchatfilter.nn.rating.MessageRating;
import java.io.File;
import java.io.IOException;
import lombok.Getter;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.deeplearning4j.eval.Evaluation;

@Log4j2
public class NetworkTrainer {
  private static final File saveLocation = new File("data/weights.zip");

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
    Network network = null;

    try {
      logger.info("Load model....");

      network = Network.loadFromFile(saveLocation);
      logger.info(network.evaluateString("Test").toString());
      logger.info(network.evaluateString("Penis").toString());

      // Number of training epochs
      final int nEpochs = 100;
      final StringIterator trainingData = new StringIterator("Test", new MessageRating(0, 0, 0));
      final StringIterator evaluationData = new StringIterator("Test", new MessageRating(0, 0, 0));

      logger.info("Train model....");

      for (int i = 0; i < nEpochs; i++) {
        network.train(trainingData);
        logger.info("*** Completed epoch {} ***", i);

        logger.info("Evaluate model....");
        Evaluation eval = network.evaluate(evaluationData);
        logger.info(eval.stats());
        evaluationData.reset();
      }

      logger.info("****************Training finished********************");
    } finally {
      if (network != null) network.saveToFile(saveLocation);

      Main.stop();
    }
  }
}
