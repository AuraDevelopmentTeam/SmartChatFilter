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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import dev.aura.smartchatfilter.util.ConsoleTests;
import java.util.Arrays;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import org.apache.commons.cli.CommandLine;
import org.junit.Test;

public class MainTest extends ConsoleTests {
  @Test
  public void noParametersTest() {
    final CommandLine result = Main.parseParameters(new String[] {});
    final int returnValue = Main.getReturnStatus();

    assertEquals(0, returnValue);
    assertNotNull(result);
    assertFalse(result.hasOption(Main.OPT_HELP));
    assertOutContentEquals("");
    assertErrContentEquals("");
    assertLogEquals(
        "[dev.aura.smartchatfilter.Main ][DEBUG]: Parsing parameters: []\n"
            + "[dev.aura.smartchatfilter.Main ][DEBUG]: Options:\n"
            + "[dev.aura.smartchatfilter.Main ][DEBUG]: Arguments: \n"
            + "[dev.aura.smartchatfilter.Main ][DEBUG]:   []");
  }

  @Test
  public void debugParameterTest() {
    final String[][] testParmeters = new String[][] {new String[] {"-d"}, new String[] {"--debug"}};

    for (final String[] parameters : testParmeters) {
      final CommandLine result = Main.parseParameters(parameters);
      final int returnValue = Main.getReturnStatus();

      assertEquals(0, returnValue);
      assertNotNull(result);
      assertOutContentEquals("");
      assertErrContentEquals("");
      assertLogEquals(
          "[dev.aura.smartchatfilter.Main ][DEBUG]: Parsing parameters: "
              + Arrays.toString(parameters)
              + "\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Options:\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]:   debug: []\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Arguments: \n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]:   []\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Debug active. Enabling TRACE logging!\n"
              + "[dev.aura.smartchatfilter.Main ][TRACE]: TRACE logging Enabled.");

      resetStreams();
    }
  }

  @Test
  public void debugVerboseParameterTest() {
    final String[][] testParmeters =
        new String[][] {
          new String[] {"-dv"},
          new String[] {"-d", "-v"},
          new String[] {"-d", "--verbose"},
          new String[] {"--debug", "-v"},
          new String[] {"--debug", "--verbose"}
        };

    for (final String[] parameters : testParmeters) {
      final CommandLine result = Main.parseParameters(parameters);
      final int returnValue = Main.getReturnStatus();

      assertEquals(0, returnValue);
      assertNotNull(result);
      assertOutContentEquals("");
      assertErrContentEquals("");
      assertLogEquals(
          "[dev.aura.smartchatfilter.Main ][DEBUG]: Parsing parameters: "
              + Arrays.toString(parameters)
              + "\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Options:\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]:   debug: []\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]:   verbose: []\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Arguments: \n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]:   []\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Verbose active. Enabling Verbose logging!\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Verbose logging Enabled.\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Debug active. Enabling TRACE logging!\n"
              + "[dev.aura.smartchatfilter.Main ][TRACE]: TRACE logging Enabled.");

      resetStreams();
    }
  }

  @Test
  public void helpParameterTest() {
    final String[][] testParmeters = new String[][] {new String[] {"-h"}, new String[] {"--help"}};

    for (final String[] parameters : testParmeters) {
      final CommandLine result = Main.parseParameters(parameters);
      final int returnValue = Main.getReturnStatus();

      assertEquals(0, returnValue);
      assertNull(result);
      assertOutContentEquals("");
      assertErrContentEquals(
          "usage: java -jar main [-d] [-h] [-v] [-V]\n"
              + "\n"
              + "SmartChatFilter: A very WIP plugin that rates and filters chat based on a neural network.\n"
              + "\n"
              + "Parameters:\n"
              + " -d,--debug     Enable more verbose logging.\n"
              + " -h,--help      Print this message.\n"
              + " -v,--verbose   Displays all messages that are printed to file.\n"
              + "                Will be very spammy in combination with -d!\n"
              + " -V,--version   Print the version.\n"
              + "\n"
              + "SmartChatFilter v0.0.1.14-DEV - A very WIP plugin that rates and filters chat based on a neural\n"
              + "network.\n"
              + "Copyright (C) 2018  Yannick Schinko\n"
              + "\n"
              + "This program is free software: you can redistribute it and/or modify\n"
              + "it under the terms of the GNU General Public License as published by\n"
              + "the Free Software Foundation, either version 3 of the License, or\n"
              + "(at your option) any later version.\n"
              + "\n"
              + "This program is distributed in the hope that it will be useful,\n"
              + "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
              + "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
              + "GNU General Public License for more details.\n"
              + "\n"
              + "You should have received a copy of the GNU General Public License\n"
              + "along with this program. If not, see <http://www.gnu.org/licenses/>.\n");
      assertLogEquals(
          "[dev.aura.smartchatfilter.Main ][DEBUG]: Parsing parameters: "
              + Arrays.toString(parameters)
              + "\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Options:\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]:   help: []\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Arguments: \n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]:   []\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Help flag detected. Printing help message and shutting down.");

      resetStreams();
    }
  }

  @Test
  public void verboseParameterTest() {
    final String[][] testParmeters =
        new String[][] {new String[] {"-v"}, new String[] {"--verbose"}};

    for (final String[] parameters : testParmeters) {
      final CommandLine result = Main.parseParameters(parameters);
      final int returnValue = Main.getReturnStatus();

      assertEquals(0, returnValue);
      assertNotNull(result);
      assertOutContentEquals("");
      assertErrContentEquals("");
      assertLogEquals(
          "[dev.aura.smartchatfilter.Main ][DEBUG]: Parsing parameters: "
              + Arrays.toString(parameters)
              + "\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Options:\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]:   verbose: []\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Arguments: \n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]:   []\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Verbose active. Enabling Verbose logging!\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Verbose logging Enabled.");

      resetStreams();
    }
  }

  @Test
  public void versionParameterTest() {
    final String[][] testParmeters =
        new String[][] {new String[] {"-V"}, new String[] {"--version"}};

    for (final String[] parameters : testParmeters) {
      final CommandLine result = Main.parseParameters(parameters);
      final int returnValue = Main.getReturnStatus();

      assertEquals(0, returnValue);
      assertNull(result);
      assertOutContentEquals("");
      assertErrContentEquals(
          "SmartChatFilter v@version@ - A very WIP plugin that rates and filters chat based on a neural network.\n"
              + "Copyright (C) 2018  Yannick Schinko\n"
              + "\n"
              + "This program is free software: you can redistribute it and/or modify\n"
              + "it under the terms of the GNU General Public License as published by\n"
              + "the Free Software Foundation, either version 3 of the License, or\n"
              + "(at your option) any later version.\n"
              + "\n"
              + "This program is distributed in the hope that it will be useful,\n"
              + "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
              + "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
              + "GNU General Public License for more details.\n"
              + "\n"
              + "You should have received a copy of the GNU General Public License\n"
              + "along with this program. If not, see <http://www.gnu.org/licenses/>.\n"
              + "\n");
      assertLogEquals(
          "[dev.aura.smartchatfilter.Main ][DEBUG]: Parsing parameters: "
              + Arrays.toString(parameters)
              + "\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Options:\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]:   version: []\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Arguments: \n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]:   []\n"
              + "[dev.aura.smartchatfilter.Main ][DEBUG]: Version flag detected. Printing version and shutting down.");

      resetStreams();
    }
  }

  @Test
  public void errorParameterTest() {
    final CommandLine result = Main.parseParameters(new String[] {"--doesnotexists"});
    final int returnValue = Main.getReturnStatus();

    assertEquals(1, returnValue);
    assertNull(result);
    assertOutContentEquals("");
    assertErrContentEquals(
        "Error: Unrecognized option: --doesnotexists\n"
            + "usage: java -jar main [-d] [-h] [-v] [-V]\n");
    assertEquals(
        "[dev.aura.smartchatfilter.Main ][DEBUG]: Parsing parameters: [--doesnotexists]",
        getLogEntry(0));
  }

  @Test
  public void normalShutdowTest() throws InterruptedException {
    Main.setReturnStatus(0);

    Main.shutdown();

    assertLogEquals(
        "[dev.aura.smartchatfilter.Main ][DEBUG]: Starting shutdown routine\n"
            + "[dev.aura.smartchatfilter.Main ][INFO ]: Application stopped.");
  }

  @Test
  public void errorShutdowTest() throws InterruptedException {
    Main.setReturnStatus(123);

    Main.shutdown();

    assertLogEquals(
        "[dev.aura.smartchatfilter.Main ][DEBUG]: Starting shutdown routine\n"
            + "[dev.aura.smartchatfilter.Main ][WARN ]: Application stopped with exit value 123.");
  }

  @Test(timeout = 600)
  public void blockTest() throws InterruptedException {
    final ScheduledThreadPoolExecutor executor = new ScheduledThreadPoolExecutor(0);
    final Thread runner = new Thread(Main::block);
    final Runnable stopperContinue = () -> Main.stop(-123);
    final Runnable interrupter = () -> runner.interrupt();
    final Runnable stopper = () -> Main.stop();

    Main.setReturnStatus(-1);
    executor.schedule(stopperContinue, 100, TimeUnit.MILLISECONDS);
    executor.schedule(stopperContinue, 200, TimeUnit.MILLISECONDS);
    executor.schedule(interrupter, 300, TimeUnit.MILLISECONDS);
    executor.schedule(stopper, 400, TimeUnit.MILLISECONDS);

    runner.start();
    runner.join();

    assertLogEquals(
        "[dev.aura.smartchatfilter.Main ][TRACE]: Return Status is -123\n"
            + "[dev.aura.smartchatfilter.Main ][TRACE]: Keep running\n"
            + "[dev.aura.smartchatfilter.Main ][TRACE]: Return Status is -123\n"
            + "[dev.aura.smartchatfilter.Main ][TRACE]: Keep running\n"
            + "[dev.aura.smartchatfilter.Main ][DEBUG]: Recived interruption:\n"
            + "java.lang.InterruptedException\n"
            + "\tat java.lang.Object.wait(Native Method)\n"
            + "\tat java.lang.Object.wait(Object.java:502)\n"
            + "\tat dev.aura.smartchatfilter.Main.block(Main.java:263)\n"
            + "\tat java.lang.Thread.run(Thread.java:748)\n"
            + "[dev.aura.smartchatfilter.Main ][TRACE]: Return Status is -123\n"
            + "[dev.aura.smartchatfilter.Main ][TRACE]: Keep running\n"
            + "[dev.aura.smartchatfilter.Main ][TRACE]: Return Status is 0\n"
            + "[dev.aura.smartchatfilter.Main ][TRACE]: Stopping");
  }
}
