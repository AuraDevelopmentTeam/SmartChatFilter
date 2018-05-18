package dev.aura.smartchatfilter.log;

import dev.aura.smartchatfilter.util.LogTests;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;

@RunWith(MockitoJUnitRunner.class)
public class AdvancedScoreIterationListenerTest extends LogTests {
  @Mock private Model model;

  @Before
  public void setup() {
    Mockito.when(model.score())
        .thenReturn(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0);
  }

  @Test
  public void defaultListenerTest() {
    final IterationListener listener = new AdvancedScoreIterationListener();

    for (int i = 0; i < 20; ++i) {
      listener.iterationDone(model, i, 20 - i);
    }

    Mockito.verify(model, Mockito.times(20)).score();
    assertLogEquals(
        "[AdvancedScoreIterationListener][INFO ]: Score at iteration 0 (epoch 20) is 1.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 1 (epoch 19) is 2.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 2 (epoch 18) is 3.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 3 (epoch 17) is 4.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 4 (epoch 16) is 5.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 5 (epoch 15) is 6.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 6 (epoch 14) is 7.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 7 (epoch 13) is 8.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 8 (epoch 12) is 9.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 9 (epoch 11) is 10.0\n"
            + "[AdvancedScoreIterationListener][INFO ]: Score at iteration 10 (epoch 10) is 11.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 11 (epoch 9) is 12.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 12 (epoch 8) is 13.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 13 (epoch 7) is 14.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 14 (epoch 6) is 15.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 15 (epoch 5) is 16.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 16 (epoch 4) is 17.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 17 (epoch 3) is 18.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 18 (epoch 2) is 19.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 19 (epoch 1) is 20.0");
  }

  @Test
  public void customListenerTest() {
    final IterationListener listener = new AdvancedScoreIterationListener(5, 2);

    for (int i = 0; i < 20; ++i) {
      listener.iterationDone(model, i, 20 - i);
    }

    Mockito.verify(model, Mockito.times(12)).score();
    assertLogEquals(
        "[AdvancedScoreIterationListener][INFO ]: Score at iteration 0 (epoch 20) is 1.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 2 (epoch 18) is 2.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 4 (epoch 16) is 3.0\n"
            + "[AdvancedScoreIterationListener][INFO ]: Score at iteration 5 (epoch 15) is 4.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 6 (epoch 14) is 5.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 8 (epoch 12) is 6.0\n"
            + "[AdvancedScoreIterationListener][INFO ]: Score at iteration 10 (epoch 10) is 7.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 12 (epoch 8) is 8.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 14 (epoch 6) is 9.0\n"
            + "[AdvancedScoreIterationListener][INFO ]: Score at iteration 15 (epoch 5) is 10.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 16 (epoch 4) is 11.0\n"
            + "[AdvancedScoreIterationListener][DEBUG]: Score at iteration 18 (epoch 2) is 12.0");
  }
}
