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
package dev.aura.smartchatfilter;

import dev.aura.smartchatfilter.util.LogTests;
import java.lang.Thread.UncaughtExceptionHandler;
import org.junit.Test;

public class UncaughtExceptionLoggerTest extends LogTests {
  @Test
  public void uncaughtExceptionTest() throws InterruptedException {
    final UncaughtExceptionHandler uncaughtExceptionHandler = new Main.UncaughtExceptionLogger();
    final Runnable runnable =
        () -> {
          throw new RuntimeException("Test Exception!");
        };
    final Thread errorThread1 = new Thread(runnable, "Error Thread 1");
    final Thread errorThread2 = new Thread(runnable, "Error Thread 2");
    errorThread1.setUncaughtExceptionHandler(uncaughtExceptionHandler);
    errorThread2.setUncaughtExceptionHandler(uncaughtExceptionHandler);

    errorThread1.start();
    errorThread1.join();
    errorThread2.start();
    errorThread2.join();

    assertLogEquals(
        "[r.Main.UncaughtExceptionLogger][FATAL]: Uncaught exception in thread \"Error Thread 1\"\n"
            + "java.lang.RuntimeException: Test Exception!\n"
            + "\tat dev.aura.smartchatfilter.UncaughtExceptionLoggerTest.lambda$uncaughtExceptionTest$0(UncaughtExceptionLoggerTest.java:30)\n"
            + "\tat java.lang.Thread.run(Thread.java:748)\n"
            + "[r.Main.UncaughtExceptionLogger][FATAL]: Uncaught exception in thread \"Error Thread 2\"\n"
            + "java.lang.RuntimeException: Test Exception!\n"
            + "\tat dev.aura.smartchatfilter.UncaughtExceptionLoggerTest.lambda$uncaughtExceptionTest$0(UncaughtExceptionLoggerTest.java:30)\n"
            + "\tat java.lang.Thread.run(Thread.java:748)");
  }
}
