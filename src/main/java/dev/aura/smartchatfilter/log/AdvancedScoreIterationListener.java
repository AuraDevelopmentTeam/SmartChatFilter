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
package dev.aura.smartchatfilter.log;

import lombok.EqualsAndHashCode;
import lombok.Generated;
import lombok.RequiredArgsConstructor;
import lombok.ToString;
import lombok.extern.log4j.Log4j2;
import org.apache.logging.log4j.Level;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

@RequiredArgsConstructor
@ToString
@EqualsAndHashCode
@Log4j2
public class AdvancedScoreIterationListener implements IterationListener {
  private static final long serialVersionUID = 825622518505976747L;

  private final int infoIterations;
  private final int debugIterations;

  @Generated
  public AdvancedScoreIterationListener(int infoIterations) {
    this(infoIterations, 1);
  }

  @Generated
  public AdvancedScoreIterationListener() {
    this(10);
  }

  @Override
  public void iterationDone(Model model, int iteration, int epoch) {
    final boolean printInfo = (iteration % infoIterations) == 0;
    final boolean printDebug = (iteration % debugIterations) == 0;

    if (printInfo || printDebug) {
      final double score = model.score();

      logger.log(
          printInfo ? Level.INFO : Level.DEBUG,
          "Score at iteration {} (epoch {}) is {}",
          iteration,
          epoch,
          score);
    }
  }
}
