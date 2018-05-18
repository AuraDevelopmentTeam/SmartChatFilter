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
package dev.aura.smartchatfilter.util;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import lombok.SneakyThrows;
import org.junit.After;
import org.junit.Before;

public class ConsoleTests extends LogTests {
  private static final String UTF_8 = StandardCharsets.UTF_8.toString();

  private final ByteArrayOutputStream outContent = new ByteArrayOutputStream();
  private final ByteArrayOutputStream errContent = new ByteArrayOutputStream();

  @SneakyThrows(UnsupportedEncodingException.class)
  protected String getOutContent() {
    return outContent.toString(UTF_8).replace("\r\n", "\n");
  }

  @SneakyThrows(UnsupportedEncodingException.class)
  protected String getErrContent() {
    return errContent.toString(UTF_8).replace("\r\n", "\n");
  }

  protected void assertOutContentEquals(String expected) {
    assertEquals(expected, getOutContent());
  }

  protected void assertErrContentEquals(String expected) {
    assertEquals(expected, getErrContent());
  }

  protected void resetStreams() {
    cleanUpStreams();
    deinitLoger();

    outContent.reset();
    errContent.reset();

    initLoger();
    setUpStreams();
  }

  @Before
  @SneakyThrows(UnsupportedEncodingException.class)
  public void setUpStreams() {
    System.setOut(new PrintStream(outContent, false, UTF_8));
    System.setErr(new PrintStream(errContent, false, UTF_8));
  }

  @After
  public void cleanUpStreams() {
    System.setOut(null);
    System.setErr(null);
  }
}
