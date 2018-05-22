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
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class StringIterator implements DataSetIterator {
  private static final long serialVersionUID = 8184624800313752541L;

  public static final int CHARACTER_COUNT = 1 << 8;
  private static final int DEFAULT_BATCH_SIZE = 64;

  private final Collection<Map.Entry<byte[], MessageRating>> originalMessages;
  private final int messagesCount;
  private final int maxLength;
  private final int miniBatchSize;

  private Iterator<Map.Entry<byte[], MessageRating>> iterator;
  private int cursor;

  private static int getByteIndex(byte val) {
    return val & 0xFF;
  }

  private static void fillMaskRow(INDArray mask, int batch, int timeStep) {
    for (int i = 0; i < CHARACTER_COUNT; ++i) {
      mask.putScalar(new int[] {batch, i, timeStep}, 1.0);
    }
  }

  public StringIterator(Map<String, MessageRating> messages) {
    this(messages, DEFAULT_BATCH_SIZE);
  }

  public StringIterator(Map<String, MessageRating> messages, int miniBatchSize) {
    this(messages.entrySet(), miniBatchSize);
  }

  public StringIterator(Collection<Map.Entry<String, MessageRating>> messages) {
    this(messages, DEFAULT_BATCH_SIZE);
  }

  public StringIterator(Collection<Map.Entry<String, MessageRating>> messages, int miniBatchSize) {
    originalMessages =
        messages
            .stream()
            .collect(
                Collectors.toMap(
                    entry -> entry.getKey().getBytes(StandardCharsets.UTF_8), Map.Entry::getValue))
            .entrySet();
    messagesCount = messages.size();
    maxLength =
        originalMessages
            .stream()
            .map(Map.Entry::getKey)
            .mapToInt(bytes -> bytes.length)
            .max()
            .getAsInt();
    this.miniBatchSize = miniBatchSize;

    reset();
  }

  @Override
  public boolean hasNext() {
    return cursor < messagesCount;
  }

  @Override
  public DataSet next() {
    return next(miniBatchSize);
  }

  @Override
  public DataSet next(int num) {
    final int examplesCount = Math.min(num, messagesCount - cursor);
    cursor += examplesCount;

    final int[] dimensions = new int[] {examplesCount, CHARACTER_COUNT, maxLength};
    final INDArray input = Nd4j.create(dimensions, 'f');
    final INDArray output = Nd4j.create(dimensions, 'f');
    final INDArray inputMask = Nd4j.create(dimensions, 'f');
    final INDArray outputMask = Nd4j.create(dimensions, 'f');

    Map.Entry<byte[], MessageRating> entry;
    byte[] string;
    int stringLength;
    MessageRating rating;

    for (int i = 0; i < examplesCount; ++i) {
      entry = iterator.next();
      string = entry.getKey();
      stringLength = string.length;
      rating = entry.getValue();

      for (int pos = 0; pos < stringLength; ++pos) {
        input.putScalar(new int[] {i, getByteIndex(string[pos]), pos}, 1.0);
        fillMaskRow(inputMask, i, pos);
      }

      output.putScalar(new int[] {i, 0, stringLength - 1}, rating.getSpam());
      output.putScalar(new int[] {i, 1, stringLength - 1}, rating.getSwearing());
      output.putScalar(new int[] {i, 2, stringLength - 1}, rating.getInsulting());
      fillMaskRow(outputMask, i, stringLength - 1);
    }

    return new DataSet(input, output, inputMask, outputMask);
  }

  @Override
  public int totalExamples() {
    return messagesCount;
  }

  @Override
  public int inputColumns() {
    return CHARACTER_COUNT;
  }

  @Override
  public int totalOutcomes() {
    return MessageRating.SCORES_COUNT;
  }

  @Override
  public boolean resetSupported() {
    return true;
  }

  @Override
  public boolean asyncSupported() {
    return false;
  }

  @Override
  public void reset() {
    iterator = originalMessages.iterator();
    cursor = 0;
  }

  @Override
  public int batch() {
    return maxLength;
  }

  @Override
  public int cursor() {
    return cursor;
  }

  @Override
  public int numExamples() {
    return totalExamples();
  }

  public void setPreProcessor(DataSetPreProcessor preProcessor) {
    throw new UnsupportedOperationException("Not implemented");
  }

  @Override
  public DataSetPreProcessor getPreProcessor() {
    throw new UnsupportedOperationException("Not implemented");
  }

  @Override
  public List<String> getLabels() {
    throw new UnsupportedOperationException("Not implemented");
  }

  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }
}
