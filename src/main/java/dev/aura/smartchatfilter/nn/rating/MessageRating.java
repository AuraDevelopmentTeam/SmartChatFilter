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
package dev.aura.smartchatfilter.nn.rating;

import lombok.Value;
import org.apache.commons.math3.exception.OutOfRangeException;

@Value
public class MessageRating {
  public static final int SCORES_COUNT = 3;

  private final double spam;
  private final double swearing;
  private final double insulting;

  private static void verifyValue(double val) throws OutOfRangeException {
    if ((val < 0.0) || (val > 1.0)) throw new OutOfRangeException(val, 0.0, 1.0);
  }

  public MessageRating(double spam, double swearing, double insulting) {
    verifyValue(spam);
    verifyValue(swearing);
    verifyValue(insulting);

    this.spam = spam;
    this.swearing = swearing;
    this.insulting = insulting;
  }
}
