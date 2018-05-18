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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.layout.PatternLayout;
import org.junit.After;
import org.junit.Before;

public class LogTests {
  private final PatternLayout LAYOUT =
      PatternLayout.newBuilder().withPattern("[%-30.30c][%-5level]: %msg\n%ex{full}").build();

  private TestLog4j2Appender appender;

  @Before
  public void initLoger() {
    final LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
    final Configuration config = ctx.getConfiguration();
    appender = TestLog4j2Appender.createAppender("TestAppender", null, LAYOUT, null);
    appender.start();
    config.addAppender(appender);

    ((org.apache.logging.log4j.core.Logger) LogManager.getRootLogger()).addAppender(appender);

    ctx.updateLoggers();
  }

  @After
  public void deinitLoger() {
    ((org.apache.logging.log4j.core.Logger) LogManager.getRootLogger()).removeAppender(appender);
    clearLogs();
  }

  protected void clearLogs() {
    appender.clearMessages();
  }

  protected String getLogEntry(int index) {
    return appender.getMessage(index);
  }

  protected String getWholeLog() {
    return appender.getAllMessages().replace("\r\n", "\n");
  }

  protected void assertLogEquals(String expected) {
    assertEquals(expected, getWholeLog());
  }
}
