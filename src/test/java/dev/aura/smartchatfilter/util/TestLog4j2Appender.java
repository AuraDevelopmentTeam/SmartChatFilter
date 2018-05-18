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

import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.logging.log4j.core.Filter;
import org.apache.logging.log4j.core.Layout;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.config.plugins.Plugin;
import org.apache.logging.log4j.core.config.plugins.PluginAttribute;
import org.apache.logging.log4j.core.config.plugins.PluginElement;
import org.apache.logging.log4j.core.config.plugins.PluginFactory;
import org.apache.logging.log4j.core.layout.PatternLayout;

// note: class name need not match the @Plugin name.
@Plugin(
  name = "TestLog4j2Appender",
  category = "Core",
  elementType = "appender",
  printObject = true
)
public final class TestLog4j2Appender extends AbstractAppender {
  private final List<String> logs = Collections.synchronizedList(new ArrayList<>());

  protected TestLog4j2Appender(
      String name,
      Filter filter,
      Layout<? extends Serializable> layout,
      final boolean ignoreExceptions) {
    super(name, filter, layout, ignoreExceptions);
  }

  // The append method is where the appender does the work.
  // Given a log event, you are free to do with it what you want.
  // This example demonstrates:
  // 1. Concurrency: this method may be called by multiple threads concurrently
  // 2. How to use layouts
  // 3. Error handling
  @Override
  public void append(LogEvent event) {
    final byte[] bytes = getLayout().toByteArray(event);

    logs.add(new String(bytes, StandardCharsets.UTF_8));
  }

  public void clearMessages() {
    logs.clear();
  }

  public String getMessage(int index) {
    if (logs.isEmpty()) {
      return "";
    }

    return logs.get(index).trim();
  }

  public String getAllMessages() {
    return String.join("", logs).trim();
  }

  // Your custom appender needs to declare a factory method
  // annotated with `@PluginFactory`. Log4j will parse the configuration
  // and call this factory method to construct an appender instance with
  // the configured attributes.
  @PluginFactory
  public static TestLog4j2Appender createAppender(
      @PluginAttribute("name") String name,
      @PluginElement("Filter") final Filter filter,
      @PluginElement("Layout") Layout<? extends Serializable> layout,
      @PluginAttribute("otherAttribute") String otherAttribute) {
    if (name == null) {
      LOGGER.error("No name provided for TestLog4j2Appender");
      return null;
    }

    if (layout == null) {
      layout = PatternLayout.createDefaultLayout();
    }

    return new TestLog4j2Appender(name, filter, layout, true);
  }
}
