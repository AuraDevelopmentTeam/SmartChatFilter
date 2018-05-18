// Copyright © 2017-2018, Project-Creative Dev-Team, All Rights Reserved
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
