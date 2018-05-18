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
public class DebugScoreIterationListener implements IterationListener {
  private static final long serialVersionUID = 825622518505976747L;

  private final int infoIterations;
  private final int debugIterations;

  @Generated
  public DebugScoreIterationListener(int infoIterations) {
    this(infoIterations, 1);
  }

  @Generated
  public DebugScoreIterationListener() {
    this(10);
  }

  @Override
  public void iterationDone(Model model, int iteration, int epoch) {
    final boolean printInfo = (iteration % infoIterations) == 0;
    final boolean printDebug = (iteration % debugIterations) == 0;

    if (printInfo || printDebug) {
      final double score = model.score();

      log.log(
          printInfo ? Level.INFO : Level.DEBUG,
          "Score at iteration {} (epoch {}) is {}",
          iteration,
          epoch,
          score);
    }
  }
}
