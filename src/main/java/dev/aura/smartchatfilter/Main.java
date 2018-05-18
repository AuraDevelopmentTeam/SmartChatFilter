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

import com.google.common.annotations.VisibleForTesting;
import dev.aura.smartchatfilter.nn.NetworkTrainer;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.lang.Thread.UncaughtExceptionHandler;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.UtilityClass;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;

@UtilityClass
@Log4j2
public class Main {
  public static final String NAME = "@name@";
  public static final String VERSION = "@version@";
  public static final String DESCRIPTION = "@description@";
  public static final String LICENSE = "@license@";

  private static final String VERSION_TEXT = LICENSE.replace(NAME, NAME + " v" + VERSION);
  private static final String HELP_HEADER = '\n' + NAME + ": " + DESCRIPTION + "\n\nParameters:";
  private static final String HELP_FOOTER = '\n' + VERSION_TEXT;

  protected static final String OPT_DEBUG = "d";
  protected static final String OPT_HELP = "h";
  protected static final String OPT_VERBOSE = "v";
  protected static final String OPT_VERSION = "V";

  @Getter @Setter private static volatile int returnStatus = -1;

  private static final Object LOCK = new Object();

  @VisibleForTesting @Getter static NetworkTrainer networkTrainer;

  private static Options getOptions() {
    final Options options = new Options();
    options.addOption(OPT_DEBUG, "debug", false, "Enable more verbose logging.");
    options.addOption(OPT_HELP, "help", false, "Print this message.");
    options.addOption(
        OPT_VERBOSE,
        "verbose",
        false,
        "Displays all messages that are printed to file.\nWill be very spammy in combination with -d!");
    options.addOption(OPT_VERSION, "version", false, "Print the version.");

    return options;
  }

  private static File getJar() {
    return new File(Main.class.getProtectionDomain().getCodeSource().getLocation().getPath());
  }

  private static void updateLogLevel(Level oldLevel, Level newLevel) {
    final LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
    final Configuration config = ctx.getConfiguration();

    config
        .getLoggers()
        .values()
        .forEach(
            logger -> {
              if (logger.getLevel() == oldLevel) {
                logger.setLevel(newLevel);
              }

              logger
                  .getAppenderRefs()
                  .stream()
                  .filter(appenderRef -> appenderRef.getLevel() == oldLevel)
                  .forEach(
                      appenderRef -> {
                        final String name = appenderRef.getRef();

                        logger.removeAppender(name);
                        logger.addAppender(
                            config.getAppender(name), newLevel, appenderRef.getFilter());
                      });
            });

    ctx.updateLoggers();
  }

  private static void enableTrace() {
    updateLogLevel(Level.DEBUG, Level.TRACE);
  }

  private static void enableVerbose(final boolean debug) {
    updateLogLevel(Level.INFO, debug ? Level.TRACE : Level.DEBUG);
  }

  protected static CommandLine parseParameters(String[] args) {
    final HelpFormatter formatter = new HelpFormatter();
    final Options options = getOptions();
    final PrintWriter stdErr =
        new PrintWriter(new OutputStreamWriter(System.err, StandardCharsets.UTF_8), true);
    final String invocation = "java -jar " + getJar().getName();
    final int terminalWidth = 100;

    try {
      logger.debug("Parsing parameters: {}", Arrays.toString(args));

      final CommandLineParser parser = new DefaultParser();
      final CommandLine commandLine = parser.parse(options, args);

      logger.debug("Options:");
      Arrays.stream(commandLine.getOptions())
          .forEach(
              option ->
                  logger.debug(
                      "  {}: {}",
                      option.hasLongOpt() ? option.getLongOpt() : option.getOpt(),
                      option.getValuesList()));
      logger.debug("Arguments: ");
      logger.debug("  {}", commandLine.getArgList());

      if (commandLine.hasOption(OPT_HELP)) {
        logger.debug("Help flag detected. Printing help message and shutting down.");

        formatter.printHelp(
            stdErr,
            terminalWidth,
            invocation,
            HELP_HEADER,
            options,
            HelpFormatter.DEFAULT_LEFT_PAD,
            HelpFormatter.DEFAULT_DESC_PAD,
            HELP_FOOTER,
            true);

        setReturnStatus(0);
        return null;
      } else if (commandLine.hasOption(OPT_VERSION)) {
        logger.debug("Version flag detected. Printing version and shutting down.");

        System.err.println(VERSION_TEXT);

        setReturnStatus(0);
        return null;
      }

      final boolean verbose = commandLine.hasOption(OPT_VERBOSE);
      final boolean debug = commandLine.hasOption(OPT_DEBUG);

      if (verbose) {
        logger.debug("Verbose active. Enabling Verbose logging!");
        enableVerbose(debug);
        logger.debug("Verbose logging Enabled.");
      }

      if (debug) {
        logger.debug("Debug active. Enabling TRACE logging!");
        enableTrace();
        logger.trace("TRACE logging Enabled.");
      }

      return commandLine;
    } catch (ParseException exp) {
      logger.debug("Error while parsing command line:", exp);

      System.err.println("Error: " + exp.getMessage());
      formatter.printUsage(stdErr, terminalWidth, invocation, options);

      setReturnStatus(1);
      return null;
    }
  }

  public static void main(String[] args) {
    try {
      logger.debug("Registering Shutdown Hook");
      Runtime.getRuntime().addShutdownHook(new Thread(Main::stop));
      logger.debug("Registering UncaughtExceptionLogger");
      Thread.setDefaultUncaughtExceptionHandler(new UncaughtExceptionLogger());

      final CommandLine commandLine = parseParameters(args);

      if (commandLine != null) {
        main(commandLine);
      }
    } catch (Exception e) {
      logger.fatal("Fatal Exception in Application:", e);

      setReturnStatus(2);
    }

    shutdown();
    LogManager.shutdown();

    System.exit(returnStatus);
  }

  protected static void main(CommandLine commandLine) throws IOException {
    try {
      logger.info("Starting " + NAME + " v" + VERSION);

      networkTrainer = new NetworkTrainer();
      networkTrainer.start();

      block();
    } catch (RuntimeException e) {
      logger.error("Error during startup: {}", e.getMessage());
      logger.debug("Detailed error:", e);
      logger.info("Check your settings!");

      setReturnStatus(1);
    }
  }

  protected static void shutdown() {
    logger.debug("Starting shutdown routine");

    if (networkTrainer != null) {
      networkTrainer.stop();
    }

    if (returnStatus == 0) {
      logger.info("Application stopped.");
    } else {
      logger.warn("Application stopped with exit value {}.", returnStatus);
    }
  }

  public static void stop() {
    stop(0);
  }

  @SuppressFBWarnings(value = "NN_NAKED_NOTIFY", justification = "State is stored in an int.")
  public static void stop(int status) {
    setReturnStatus(status);

    synchronized (LOCK) {
      LOCK.notify();
    }
  }

  protected static void block() {
    synchronized (LOCK) {
      boolean printMessage = false;

      while (keepRunning(printMessage)) {
        try {
          LOCK.wait();
        } catch (InterruptedException e) {
          logger.debug("Recived interruption:", e);
        }

        printMessage = true;
      }
    }
  }

  private static boolean keepRunning(boolean verbose) {
    final boolean run = returnStatus < 0;

    if (verbose && logger.isTraceEnabled()) {
      logger.trace("Return Status is {}", returnStatus);

      if (run) {
        logger.trace("Keep running");
      } else {
        logger.trace("Stopping");
      }
    }

    return run;
  }

  @Log4j2
  protected static class UncaughtExceptionLogger implements UncaughtExceptionHandler {
    @Override
    public void uncaughtException(Thread thread, Throwable exception) {
      logger.fatal("Uncaught exception in thread \"" + thread.getName() + '"', exception);
    }
  }
}
