// Submission agent package declaration
package ai.abstraction.submissions.parker_riggs;

import ai.abstraction.LightRush;
import ai.abstraction.WorkerRush;
import ai.core.AI;
import ai.core.ParameterSpecification;
import java.util.ArrayList;
import java.util.List;
import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;

public class AdaptiveRushBot extends AI {
    private static final int LIGHT_RUSH_TIME_THRESHOLD = 300;
    private static final int WORKER_COUNT_THRESHOLD = 6;
    private static final int LIGHT_COUNT_THRESHOLD = 2;
    private static final int ENEMY_PRESSURE_DISTANCE = 8;

    private final UnitTypeTable unitTypeTable;
    private final WorkerRush workerRush;
    private final LightRush lightRush;

    public AdaptiveRushBot(UnitTypeTable unitTypeTable) {
        this.unitTypeTable = unitTypeTable;
        this.workerRush = new WorkerRush(unitTypeTable);
        this.lightRush = new LightRush(unitTypeTable);
    }

    @Override
    public void reset() {
        workerRush.reset();
        lightRush.reset();
    }

    @Override
    public AI clone() {
        return new AdaptiveRushBot(unitTypeTable);
    }

    @Override
    public PlayerAction getAction(int player, GameState gameState) throws Exception {
        if (gameState.canExecuteAnyAction(player)) {
            if (shouldUseLightRush(player, gameState)) {
                return lightRush.getAction(player, gameState);
            }
            return workerRush.getAction(player, gameState);
        }
        return new PlayerAction();
    }

    private boolean shouldUseLightRush(int player, GameState gameState) {
        if (gameState.getTime() >= LIGHT_RUSH_TIME_THRESHOLD) {
            return true;
        }

        PhysicalGameState physicalGameState = gameState.getPhysicalGameState();
        int workers = 0;
        int lightUnits = 0;
        List<Unit> ownBases = new ArrayList<>();
        int closestEnemyDistanceToBase = Integer.MAX_VALUE;

        for (Unit unit : physicalGameState.getUnits()) {
            UnitType type = unit.getType();
            String typeName = type.name;

            if (unit.getPlayer() == player) {
                if ("Worker".equals(typeName)) {
                    workers++;
                } else if ("Light".equals(typeName)) {
                    lightUnits++;
                } else if ("Base".equals(typeName)) {
                    ownBases.add(unit);
                }
            }
        }

        if (lightUnits >= LIGHT_COUNT_THRESHOLD || workers >= WORKER_COUNT_THRESHOLD) {
            return true;
        }

        if (ownBases.isEmpty()) {
            return true;
        }

        for (Unit unit : physicalGameState.getUnits()) {
            if (unit.getPlayer() < 0 || unit.getPlayer() == player) {
                continue;
            }

            if (!unit.getType().canAttack) {
                continue;
            }

            for (Unit base : ownBases) {
                int distance = Math.abs(unit.getX() - base.getX()) + Math.abs(unit.getY() - base.getY());
                if (distance < closestEnemyDistanceToBase) {
                    closestEnemyDistanceToBase = distance;
                }
            }
        }

        return closestEnemyDistanceToBase <= ENEMY_PRESSURE_DISTANCE;
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}