/*
 * Template agent for MicroRTS LLM Competition.
 *
 * Instructions:
 * 1. Copy this folder to submissions/your-team-name/
 * 2. Rename this file to match your agent_class in metadata.json
 * 3. Update the package name below (replace "your_team_name" with your folder name, hyphens → underscores)
 * 4. Implement your strategy in getAction()
 * 5. Update metadata.json with your team info
 */
package ai.abstraction.submissions.parker-riggs;

import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import java.util.ArrayList;
import java.util.List;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.*;

public class Agent extends AbstractionLayerAI {

    protected UnitTypeTable utt;

    // Required constructor - must accept UnitTypeTable
    public Agent(UnitTypeTable a_utt) {
        super(new AStarPathFinding());
        reset(a_utt);
    }

    public void reset() {
        super.reset();
    }

    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
    }

    public AI clone() {
        return new Agent(utt);
    }

    /*
     * Main entry point - called every game tick.
     *
     * Available high-level actions (from AbstractionLayerAI):
     *   move(unit, x, y)           - Move unit to position
     *   attack(unit, target)       - Attack a target unit
     *   harvest(unit, resource, base) - Harvest resource and return to base
     *   build(unit, type, x, y)    - Build a structure at position
     *   train(unit, type)          - Train a unit from a building
     *   idle(unit)                 - Do nothing this tick
     *
     * Must return translateActions(player, gs) at the end.
     */
    public PlayerAction getAction(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player p = gs.getPlayer(player);

        // TODO: Implement your strategy here
        // Example: iterate over your units and assign actions
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player && gs.getActionAssignment(u) == null) {
                idle(u);
            }
        }

        return translateActions(player, gs);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}
