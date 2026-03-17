package ai.mcts.submissions.nick_mcts; 

import ai.abstraction.WorkerRush;
import ai.core.AI;
import ai.core.ParameterSpecification;
import ai.evaluation.LanchesterEvaluationFunction;
import ai.mcts.naivemcts.NaiveMCTS;
import java.util.ArrayList;
import java.util.List;
import rts.GameState;
import rts.units.UnitTypeTable;
import rts.units.Unit;
import rts.PhysicalGameState;

public class NickMCTS extends NaiveMCTS {
    private UnitTypeTable utt;

    public NickMCTS(UnitTypeTable utt) {
        super(100, -1, 100, 10,
              0.3f, 0.0f, 0.4f,
              new WorkerRush(utt), 
              new MyEvaluation(utt), 
              true);
        this.utt = utt;
    }

    @Override
    public AI clone() {
        return new NickMCTS(utt); // Pass the actual UTT, not null
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}

class MyEvaluation extends LanchesterEvaluationFunction {
    private UnitTypeTable utt;

    public MyEvaluation(UnitTypeTable utt) {
        this.utt = utt;
    }

    @Override
    public float evaluate(int maxplayer, int minplayer, GameState gs) {
        float score = super.evaluate(maxplayer, minplayer, gs);
        float threatPenalty = calculateThreat(maxplayer, gs);
        
        float carryingBonus = 0;
        for (Unit u : gs.getUnits()) {
            if (u.getPlayer() == maxplayer && u.getResources() > 0) {
                carryingBonus += 0.2f;
            }
        }
        return score - threatPenalty + carryingBonus;
    }

    private float calculateThreat(int player, GameState gs) {
        float threatPenalty = 0.0f;
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int enemy = 1 - player;

        // 1. Identify Bases and Enemy Units once
        List<Unit> myBases = new ArrayList<>();
        List<Unit> enemyUnits = new ArrayList<>();

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                // Use type ID or check name once
                if (u.getType().name.equals("Base")) myBases.add(u);
            } else if (u.getPlayer() == enemy) {
                enemyUnits.add(u);
            }
        }

        // 2. If no bases, no threat to calculate
        if (myBases.isEmpty()) return 0.0f;

        // 3. Only calculate distances between bases and enemies
        for (Unit base : myBases) {
            int bx = base.getX();
            int by = base.getY();
            for (Unit e : enemyUnits) {
                // Manhattan distance
                int dist = Math.abs(bx - e.getX()) + Math.abs(by - e.getY());
                if (dist < 8) {
                    threatPenalty += (8 - dist) * 0.1f;
                }
            }
        }
        return threatPenalty;
    }
}

