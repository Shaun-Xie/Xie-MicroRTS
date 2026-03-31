package ai.mcts.submissions.xiebot;

import ai.RandomBiasedAI;
import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import ai.evaluation.SimpleSqrtEvaluationFunction3;
import ai.mcts.naivemcts.NaiveMCTS;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;

/**
 * XieBot v5
 *
 * Simplified controller:
 * - PPO-style policy inference chooses one of four high-leverage macro modes on a slow schedule.
 * - Scripted logic executes production, economy, and movement cheaply every tick.
 * - NaiveMCTS is used only as a tactical heuristic during dense local fights and base defense.
 *
 * The bot is fully self-contained: no runtime Python process, network calls, or API usage.
 * Exported policy weights must follow the fixed 25 -> 12 -> 8 -> 4 layout.
 */
public class XieBot extends AbstractionLayerAI {
    private enum DecisionSource {
        OPENING,
        POLICY,
        FALLBACK,
        EMERGENCY
    }

    private enum OpeningPlan {
        ANTI_WORKER_RUSH,
        ANTI_LIGHT_RUSH,
        ANTI_HEAVY_RUSH,
        STANDARD_OPENING,
        LARGE_MAP_BOOM
    }

    // ===== Core unit types =====
    private UnitTypeTable utt;
    private UnitType workerType;
    private UnitType baseType;
    private UnitType barracksType;
    private UnitType lightType;
    private UnitType rangedType;
    private UnitType heavyType;

    // ===== PPO macro action space =====
    private enum MacroAction {
        DEFEND_RUSH,
        STANDARD_PRESSURE,
        RANGED_TRANSITION,
        ECON_BOOM;

        boolean prefersPressure() {
            return this == STANDARD_PRESSURE;
        }

        boolean prefersDefense() {
            return this == DEFEND_RUSH;
        }

        boolean prefersRanged() {
            return this == RANGED_TRANSITION;
        }

        boolean prefersEconomy() {
            return this == ECON_BOOM;
        }

        boolean attacksWhenStable() {
            return this == STANDARD_PRESSURE || this == RANGED_TRANSITION;
        }
    }

    private static final MacroAction[] MACRO_ACTIONS = MacroAction.values();

    // ===== Embedded PPO policy config =====
    private static final int FEATURE_COUNT = 25;
    private static final int POLICY_HIDDEN_1 = 12;
    private static final int POLICY_HIDDEN_2 = 8;
    private static final int ACTION_COUNT = MACRO_ACTIONS.length;
    private static final int POLICY_DECISION_INTERVAL = 96;
    private static final int POLICY_LOCK_TICKS = 128;
    private static final int RECENT_COMBAT_WINDOW = 96;
    private static final int SMALL_MAP_OPENING_TICKS = 700;
    private static final int LARGE_MAP_OPENING_TICKS = 900;
    private static final float POLICY_MIN_CONFIDENCE = 0.34f;
    private static final float POLICY_MIN_MARGIN = 0.08f;
    private static final boolean DEBUG = Boolean.getBoolean("xiebot.debug");

    // ===== Exported PPO policy weights =====
    // Export layout must match the fixed 4-action order:
    // DEFEND_RUSH, STANDARD_PRESSURE, RANGED_TRANSITION, ECON_BOOM

private static final float[][] PPO_W1 = new float[][]{
    {-0.03201695f, 0.02389964f, -0.36004615f, -0.38419503f, -0.46816945f, -0.51148880f, -0.26986346f, 0.09862339f, -0.35981795f, 0.02413375f, 0.01903264f, -0.42761990f},
    {0.15847336f, -0.07443745f, -0.20181148f, -0.08944098f, -0.10585838f, 0.12224854f, -0.03058811f, 0.11835890f, -0.30201510f, 0.15386276f, 0.05549263f, 0.08375218f},
    {-0.30265978f, -0.01495448f, 0.16629937f, 0.33811471f, 0.29952061f, 0.22168261f, -0.26349795f, -0.30070078f, 0.31218255f, -0.34703681f, -0.18066071f, 0.14578184f},
    {0.23062873f, 0.07075119f, -0.23474804f, -0.15556486f, 0.08688487f, 0.23946291f, 0.07549202f, 0.26300457f, -0.09025887f, 0.13747306f, 0.30470592f, 0.20407219f},
    {-0.03843627f, 0.16003333f, -0.36035663f, -0.58089644f, -0.47033611f, -0.31530195f, -0.30152503f, 0.10970266f, -0.46511719f, 0.16469191f, 0.14794312f, -0.60128456f},
    {0.15368550f, -0.03706460f, -0.34936774f, -0.48179418f, -0.28646627f, -0.20379843f, 0.01827345f, 0.26925552f, -0.50615668f, 0.21670888f, 0.22845580f, -0.45024514f},
    {-0.02510832f, -0.00778295f, -0.13012388f, -0.26560694f, -0.28013429f, -0.30017236f, -0.30711630f, 0.16371226f, -0.35087833f, 0.05075114f, 0.04016818f, -0.35331103f},
    {0.19896841f, -0.08912537f, -0.10115449f, -0.10416142f, 0.16637528f, 0.34356129f, 0.18931368f, 0.28806573f, -0.11320275f, 0.29508650f, 0.20629962f, 0.22238390f},
    {-0.38802421f, 0.02773440f, 0.13958043f, 0.00949504f, -0.12719582f, -0.15852480f, -0.19374001f, -0.12459238f, 0.19437039f, -0.29895544f, -0.28419229f, -0.08322747f},
    {-0.42631412f, -0.01647443f, 0.17711587f, 0.08781225f, -0.25907066f, -0.24683814f, -0.28554535f, -0.46001869f, 0.15691239f, -0.55874616f, -0.33968395f, -0.20720783f},
    {-0.66880882f, -0.04728227f, 0.47957963f, 0.21931858f, -0.15352808f, -0.33674446f, -0.75375211f, -0.58889329f, 0.34236598f, -0.61174917f, -0.70291102f, -0.17506336f},
    {0.02932955f, -0.12294680f, 0.23691073f, 0.28646588f, 0.42187294f, 0.44117036f, -0.04748769f, -0.01962197f, 0.18326862f, -0.00572757f, 0.01610500f, 0.42154470f},
    {0.08170914f, -0.10221801f, 0.15930519f, 0.08357475f, 0.52462649f, 0.57538664f, 0.01687719f, 0.12047147f, -0.05816217f, 0.13449028f, 0.10229778f, 0.53958434f},
    {0.03644435f, -0.06454767f, 0.16362278f, -0.02435532f, 0.07809260f, 0.46749553f, 0.31717005f, 0.05924666f, 0.00709970f, 0.08367819f, 0.00979481f, 0.35539803f},
    {-0.02267172f, 0.00571535f, 0.18904734f, 0.22998874f, 0.28326213f, 0.56659889f, 0.03548321f, -0.09770085f, 0.30144471f, -0.10719223f, 0.00210958f, 0.47311959f},
    {0.10127424f, -0.04063688f, -0.23389915f, -0.25931692f, -0.00984239f, -0.03952233f, 0.02996833f, 0.20950754f, -0.30243304f, 0.16445608f, 0.03238981f, 0.18482205f},
    {0.55230057f, -0.01125667f, -0.57343525f, -0.45849857f, 0.07291926f, 0.06628775f, 0.44729671f, 0.66371876f, -0.68823093f, 0.66710627f, 0.58031982f, 0.09630738f},
    {0.41403711f, 0.11875564f, -0.44622946f, -0.49758410f, 0.07031149f, -0.07599695f, 0.16281711f, 0.35677865f, -0.51103151f, 0.26488727f, 0.33871308f, -0.01476461f},
    {0.22896831f, 0.00308278f, -0.55080849f, -0.55078524f, -0.18054768f, -0.54293621f, -0.00187068f, 0.36035702f, -0.48291439f, 0.34757224f, 0.22014187f, -0.35132509f},
    {0.37931004f, -0.07093135f, -0.46915251f, -0.52019113f, -0.04978148f, 0.08147376f, 0.31707621f, 0.36957651f, -0.42736238f, 0.49793804f, 0.41489643f, -0.10380308f},
    {-0.36026016f, 0.05945578f, 0.23143369f, 0.16589291f, -0.06258313f, 0.04733161f, -0.15436345f, -0.47935629f, 0.17506775f, -0.50509864f, -0.56653297f, -0.09599257f},
    {0.14248978f, 0.00376959f, -0.07195212f, -0.06977081f, 0.11413642f, -0.00624908f, -0.02330985f, 0.03879537f, 0.00567506f, 0.11456390f, 0.08189324f, 0.01760130f},
    {-0.00193229f, 0.04088870f, -0.09238309f, 0.05320313f, -0.09639032f, -0.00130866f, 0.02888636f, 0.00048456f, 0.08022503f, 0.11604743f, 0.02510346f, 0.05711210f},
    {0.03738160f, -0.20908476f, 0.04711387f, 0.02101590f, 0.12640852f, 0.02086297f, -0.04649977f, 0.02875594f, 0.10922825f, 0.04831474f, 0.01008672f, 0.24113834f},
    {0.01440021f, -0.03115442f, -0.14983790f, 0.10358918f, 0.08473469f, 0.05531862f, 0.04971097f, 0.07369699f, -0.00680807f, 0.05308900f, 0.04421225f, 0.00517998f}
};

private static final float[] PPO_B1 = new float[]{0.01786213f, 0.00000000f, 0.03078385f, 0.02764079f, 0.10782601f, 0.09013186f, -0.02025042f, 0.05433403f, 0.01778700f, 0.03731877f, 0.01924683f, 0.10374275f};

private static final float[][] PPO_W2 = new float[][]{
    {0.18417680f, -0.54683459f, 0.07078879f, -0.06038831f, -0.61250663f, 0.24785978f, -0.54210401f, 0.07354746f},
    {-0.02769455f, -0.08872427f, -0.00534892f, 0.06989264f, -0.03140298f, -0.01817941f, -0.01768275f, 0.00876755f},
    {-0.80195546f, 0.08582719f, -0.74868262f, 0.07537419f, 0.02530822f, -0.62496620f, 0.21749656f, -0.70798755f},
    {-0.52970272f, 0.13098440f, -0.47325712f, -0.07522067f, 0.12910198f, -0.31268805f, 0.10346507f, -0.56456274f},
    {0.08814735f, 0.30422646f, 0.04237057f, -0.09798340f, 0.36170316f, 0.15695806f, 0.34762993f, 0.23566146f},
    {0.27064273f, 0.31663042f, 0.37557110f, -0.02187353f, 0.22657748f, 0.15144452f, 0.23551123f, 0.29711631f},
    {0.35421833f, -0.54216027f, 0.26313820f, -0.05216269f, -0.42106998f, 0.27582976f, -0.47983304f, 0.33400759f},
    {-0.00170825f, -0.64507222f, 0.04457243f, -0.02156335f, -0.56869531f, 0.22390021f, -0.57646316f, 0.02119452f},
    {-0.65905005f, 0.05297268f, -0.69276470f, -0.10804359f, 0.07363366f, -0.48618788f, 0.02520275f, -0.73331207f},
    {0.00830581f, -0.72597623f, 0.10238249f, 0.16510084f, -0.62530810f, 0.02520561f, -0.76055032f, 0.02723595f},
    {0.04580354f, -0.74168259f, 0.08525670f, -0.10696840f, -0.51839036f, 0.18167369f, -0.54420602f, 0.00128378f},
    {0.28539562f, 0.24637495f, 0.18108492f, -0.02960830f, 0.15628320f, 0.14101402f, 0.28943664f, 0.18440033f}
};

private static final float[] PPO_B2 = new float[]{0.08204690f, 0.09804151f, 0.09286598f, -0.01138246f, 0.11282070f, 0.11820723f, 0.05818675f, 0.06166885f};

private static final float[][] PPO_W3 = new float[][]{
    {0.01669501f, 0.02678507f, 0.00335847f, -0.23682162f},
    {0.14406532f, -0.20880899f, 0.26114619f, -0.33068180f},
    {0.11660985f, 0.08709146f, -0.25587907f, -0.12998858f},
    {0.07536386f, 0.06221622f, -0.07636169f, -0.00126753f},
    {0.12672409f, -0.19589292f, 0.04461608f, -0.15373135f},
    {0.01309221f, 0.03138942f, -0.15717024f, -0.19790195f},
    {0.09179147f, -0.16930315f, 0.13588597f, -0.33486354f},
    {0.03000853f, -0.01068947f, -0.07855564f, -0.26982901f}
};

private static final float[] PPO_B3 = new float[]{0.09567998f, -0.06419884f, 0.02870183f, -0.12045155f};

    // ===== Tactical MCTS config =====
    private final NaiveMCTS tacticalMCTS;
    private static final int MCTS_TIME_BUDGET_MS = 35;
    private static final int MCTS_LOOKAHEAD = 80;
    private static final int MCTS_MAX_DEPTH = 8;
    private static final int DEFENSE_RADIUS = 6;

    // ===== Runtime state =====
    private OpeningPlan currentOpeningPlan = OpeningPlan.STANDARD_OPENING;
    private MacroAction currentMacroAction = MacroAction.STANDARD_PRESSURE;
    private int lastOpeningDecisionTime = -9999;
    private int lastPolicyEvaluationTime = -9999;
    private int lastStrategyChangeTime = -9999;
    private int lastCombatTime = -9999;
    private float lastPolicyConfidence = 0.0f;
    private float lastPolicyMargin = 0.0f;
    private DecisionSource lastDecisionSource = DecisionSource.FALLBACK;
    private int policyDecisionCount = 0;
    private int openingDecisionCount = 0;
    private int fallbackDecisionCount = 0;
    private int emergencyDecisionCount = 0;
    private int mctsAttemptCount = 0;
    private int mctsSuccessCount = 0;
    private int firstEnemyBarracksSeenTime = -1;
    private int firstEnemyMilitarySeenTime = -1;
    private int firstOwnBarracksSeenTime = -1;
    private int firstDefenderSeenTime = -1;
    private int firstMctsAttemptTime = -1;
    private int firstMctsSuccessTime = -1;
    private int openingStartWorkerCount = -1;
    private int openingWorkerLosses = 0;
    private int lastDebugDecisionTime = -9999;

    public XieBot(UnitTypeTable a_utt) {
        super(new AStarPathFinding());
        tacticalMCTS = new NaiveMCTS(
                MCTS_TIME_BUDGET_MS,
                -1,
                MCTS_LOOKAHEAD,
                MCTS_MAX_DEPTH,
                0.25f,
                0.0f,
                0.35f,
                new RandomBiasedAI(),
                new SimpleSqrtEvaluationFunction3(),
                true
        );
        reset(a_utt);
    }

    @Override
    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
        workerType = utt.getUnitType("Worker");
        baseType = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
        lightType = utt.getUnitType("Light");
        rangedType = utt.getUnitType("Ranged");
        heavyType = utt.getUnitType("Heavy");
        reset();
    }

    @Override
    public void reset() {
        super.reset();
        currentOpeningPlan = OpeningPlan.STANDARD_OPENING;
        currentMacroAction = MacroAction.STANDARD_PRESSURE;
        lastOpeningDecisionTime = -9999;
        lastPolicyEvaluationTime = -9999;
        lastStrategyChangeTime = -9999;
        lastCombatTime = -9999;
        lastPolicyConfidence = 0.0f;
        lastPolicyMargin = 0.0f;
        lastDecisionSource = DecisionSource.FALLBACK;
        policyDecisionCount = 0;
        openingDecisionCount = 0;
        fallbackDecisionCount = 0;
        emergencyDecisionCount = 0;
        mctsAttemptCount = 0;
        mctsSuccessCount = 0;
        firstEnemyBarracksSeenTime = -1;
        firstEnemyMilitarySeenTime = -1;
        firstOwnBarracksSeenTime = -1;
        firstDefenderSeenTime = -1;
        firstMctsAttemptTime = -1;
        firstMctsSuccessTime = -1;
        openingStartWorkerCount = -1;
        openingWorkerLosses = 0;
        lastDebugDecisionTime = -9999;
        tacticalMCTS.reset();
    }

    @Override
    public AI clone() {
        return new XieBot(utt);
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) {
        if (!gs.canExecuteAnyAction(player)) {
            return new PlayerAction();
        }

        PhysicalGameState pgs = gs.getPhysicalGameState();
        int enemy = 1 - player;

        Counts my = countUnits(pgs, player);
        Counts opp = countUnits(pgs, enemy);

        boolean smallMap = pgs.getWidth() * pgs.getHeight() <= 100;
        boolean opening = isOpeningPhase(gs, smallMap);
        float basePressure = basePressureIndicator(pgs, my, enemy);
        boolean enemyNearBase = isEnemyNearBase(pgs, my, enemy, 4) || basePressure >= 0.60f;
        updateOpeningTelemetry(gs.getTime(), my, opp, opening);

        boolean combatNow = isCombatHappening(pgs, my, enemy);
        if (combatNow) {
            lastCombatTime = gs.getTime();
        }
        boolean recentCombat = combatNow || gs.getTime() - lastCombatTime <= RECENT_COMBAT_WINDOW;

        if (opening) {
            maybeUpdateOpeningPlan(gs, my, opp, smallMap, enemyNearBase, basePressure);
        } else {
            maybeUpdateMacroAction(gs, player, my, opp, smallMap, enemyNearBase, recentCombat, basePressure);
        }

        if (shouldUseMCTS(gs, my, opp, opening, enemyNearBase, smallMap, recentCombat)) {
            mctsAttemptCount++;
            if (firstMctsAttemptTime < 0) {
                firstMctsAttemptTime = gs.getTime();
            }
            try {
                PlayerAction mctsAction = tacticalMCTS.getAction(player, gs);
                if (mctsAction != null && !mctsAction.isEmpty()) {
                    mctsSuccessCount++;
                    if (firstMctsSuccessTime < 0) {
                        firstMctsSuccessTime = gs.getTime();
                    }
                    return mctsAction;
                }
            } catch (Exception ignored) {
                // Tactical inference is optional. Fall through to scripted control immediately.
            }
        }

        applyMacroStrategy(gs, player, my, opp, enemyNearBase, smallMap, opening);
        return translateActions(player, gs);
    }

    // ===========================
    // Hierarchical macro control
    // ===========================
    private void maybeUpdateOpeningPlan(
            GameState gs,
            Counts my,
            Counts opp,
            boolean smallMap,
            boolean enemyNearBase,
            float basePressure) {

        OpeningPlan next = detectOpeningPlan(gs, my, opp, smallMap, enemyNearBase, basePressure);
        currentMacroAction = mapOpeningPlanToMacro(next);
        lastDecisionSource = DecisionSource.OPENING;
        lastPolicyConfidence = 0.0f;
        lastPolicyMargin = 0.0f;

        if (next != currentOpeningPlan || lastOpeningDecisionTime < 0) {
            currentOpeningPlan = next;
            openingDecisionCount++;
            lastOpeningDecisionTime = gs.getTime();
            if (DEBUG) {
                System.err.printf(
                        "XieBot[t=%d] opening=%s macro=%s intel(enemyBarracks=%d enemyMilitary=%d ownBarracks=%d defender=%d mcts=%d/%d@%d/%d workerLosses=%d)%n",
                        gs.getTime(),
                        currentOpeningPlan,
                        currentMacroAction,
                        firstEnemyBarracksSeenTime,
                        firstEnemyMilitarySeenTime,
                        firstOwnBarracksSeenTime,
                        firstDefenderSeenTime,
                        mctsSuccessCount,
                        mctsAttemptCount,
                        firstMctsAttemptTime,
                        firstMctsSuccessTime,
                        openingWorkerLosses
                );
            }
        }
    }

    private OpeningPlan detectOpeningPlan(
            GameState gs,
            Counts my,
            Counts opp,
            boolean smallMap,
            boolean enemyNearBase,
            float basePressure) {

        boolean enemyBarracksSeen = !opp.barracks.isEmpty();
        boolean lowEnemyEconomy = opp.workers.size() <= 1;
        boolean workerAggro = !enemyBarracksSeen
                && opp.workers.size() >= 3
                && (enemyNearBase || basePressure >= 0.45f);

        if (!opp.heavy.isEmpty()) {
            return OpeningPlan.ANTI_HEAVY_RUSH;
        }
        if (!opp.light.isEmpty()) {
            return OpeningPlan.ANTI_LIGHT_RUSH;
        }
        if (workerAggro) {
            return OpeningPlan.ANTI_WORKER_RUSH;
        }
        if (currentOpeningPlan == OpeningPlan.ANTI_WORKER_RUSH
                || currentOpeningPlan == OpeningPlan.ANTI_LIGHT_RUSH
                || currentOpeningPlan == OpeningPlan.ANTI_HEAVY_RUSH) {
            return currentOpeningPlan;
        }
        if (smallMap && enemyBarracksSeen && lowEnemyEconomy && gs.getTime() <= 360) {
            return OpeningPlan.ANTI_LIGHT_RUSH;
        }
        if (!smallMap && !enemyBarracksSeen && opp.militaryCount() == 0 && !enemyNearBase) {
            return OpeningPlan.LARGE_MAP_BOOM;
        }
        return smallMap ? OpeningPlan.STANDARD_OPENING : OpeningPlan.LARGE_MAP_BOOM;
    }

    private MacroAction mapOpeningPlanToMacro(OpeningPlan openingPlan) {
        switch (openingPlan) {
            case ANTI_WORKER_RUSH:
            case ANTI_LIGHT_RUSH:
            case ANTI_HEAVY_RUSH:
                return MacroAction.DEFEND_RUSH;
            case LARGE_MAP_BOOM:
                return MacroAction.ECON_BOOM;
            default:
                return MacroAction.STANDARD_PRESSURE;
        }
    }

    private void maybeUpdateMacroAction(
            GameState gs,
            int player,
            Counts my,
            Counts opp,
            boolean smallMap,
            boolean enemyNearBase,
            boolean recentCombat,
            float basePressure) {

        boolean emergency = isEmergencyState(gs, player, my, opp, enemyNearBase, recentCombat, basePressure);
        int now = gs.getTime();

        if (!emergency && now - lastPolicyEvaluationTime < POLICY_DECISION_INTERVAL) {
            return;
        }
        if (!emergency && now - lastStrategyChangeTime < POLICY_LOCK_TICKS) {
            return;
        }

        lastPolicyEvaluationTime = now;
        MacroAction next = selectMacroAction(gs, player, my, opp, smallMap, enemyNearBase, recentCombat, basePressure, emergency);
        recordDecision(now, next);

        if (emergency) {
            currentMacroAction = next;
            lastStrategyChangeTime = now;
            return;
        }

        if (next != currentMacroAction) {
            currentMacroAction = next;
            lastStrategyChangeTime = now;
        } else if (lastStrategyChangeTime < 0) {
            lastStrategyChangeTime = now;
        }
    }

    private MacroAction selectMacroAction(
            GameState gs,
            int player,
            Counts my,
            Counts opp,
            boolean smallMap,
            boolean enemyNearBase,
            boolean recentCombat,
            float basePressure,
            boolean emergency) {

        if (emergency) {
            lastPolicyConfidence = 0.0f;
            lastPolicyMargin = 0.0f;
            lastDecisionSource = DecisionSource.EMERGENCY;
            return runFallbackStrategy(gs, player, my, opp, smallMap, enemyNearBase, recentCombat);
        }

        try {
            float[] features = extractFeatures(gs, player, my, opp, enemyNearBase, recentCombat, basePressure);
            PolicyOutput output = forwardPolicy(features);
            lastPolicyConfidence = output.confidence;
            lastPolicyMargin = output.margin;

            if (output.confidence >= POLICY_MIN_CONFIDENCE && output.margin >= POLICY_MIN_MARGIN) {
                lastDecisionSource = DecisionSource.POLICY;
                return output.action;
            }
        } catch (Exception ignored) {
            lastPolicyConfidence = 0.0f;
            lastPolicyMargin = 0.0f;
        }

        lastDecisionSource = DecisionSource.FALLBACK;
        return runFallbackStrategy(gs, player, my, opp, smallMap, enemyNearBase, recentCombat);
    }

    private void recordDecision(int now, MacroAction next) {
        switch (lastDecisionSource) {
            case POLICY:
                policyDecisionCount++;
                break;
            case EMERGENCY:
                emergencyDecisionCount++;
                break;
            default:
                fallbackDecisionCount++;
                break;
        }

        if (!DEBUG) {
            return;
        }

        boolean macroChanged = next != currentMacroAction;
        boolean firstDecision = lastDebugDecisionTime < 0;
        if (firstDecision || macroChanged || now - lastDebugDecisionTime >= POLICY_DECISION_INTERVAL * 3) {
            lastDebugDecisionTime = now;
            System.err.printf(
                    "XieBot[t=%d] macro=%s source=%s conf=%.3f margin=%.3f decisions(policy=%d fallback=%d emergency=%d) mcts=%d/%d%n",
                    now,
                    next,
                    lastDecisionSource,
                    lastPolicyConfidence,
                    lastPolicyMargin,
                    policyDecisionCount,
                    fallbackDecisionCount,
                    emergencyDecisionCount,
                    mctsSuccessCount,
                    mctsAttemptCount
            );
        }
    }

    private void applyMacroStrategy(
            GameState gs,
            int player,
            Counts my,
            Counts opp,
            boolean enemyNearBase,
            boolean smallMap,
            boolean opening) {

        Unit myBase = my.bases.isEmpty() ? null : my.bases.get(0);
        Unit enemyBase = opp.bases.isEmpty() ? null : opp.bases.get(0);

        handleBaseProduction(gs, player, my, opp, enemyNearBase, smallMap, opening);
        handleBarracksProduction(gs, player, my, opp, enemyNearBase, smallMap, opening);
        handleWorkers(gs, player, my, opp, myBase, enemyBase, enemyNearBase, smallMap, opening);
        handleMilitary(gs, 1 - player, my, opp, myBase, enemyBase, enemyNearBase, opening);
    }

    private boolean isEmergencyState(
            GameState gs,
            int player,
            Counts my,
            Counts opp,
            boolean enemyNearBase,
            boolean recentCombat,
            float basePressure) {

        Player me = gs.getPlayer(player);
        if (my.bases.isEmpty()) {
            return true;
        }
        if (my.workers.isEmpty()) {
            return true;
        }
        if (enemyNearBase
                && basePressure >= 0.70f
                && my.combatWithWorkers() + 1 < opp.combatWithWorkers()) {
            return true;
        }
        if (my.workers.size() <= 1
                && my.barracks.isEmpty()
                && me.getResources() < barracksType.cost) {
            return true;
        }
        return recentCombat && my.militaryCount() == 0 && opp.militaryCount() > 0;
    }

    private MacroAction runFallbackStrategy(
            GameState gs,
            int player,
            Counts my,
            Counts opp,
            boolean smallMap,
            boolean enemyNearBase,
            boolean recentCombat) {

        Player me = gs.getPlayer(player);

        boolean opening = isOpeningPhase(gs, smallMap);

        if (my.bases.isEmpty()) {
            return my.militaryCount() >= Math.max(2, opp.militaryCount() - 1)
                    ? MacroAction.STANDARD_PRESSURE
                    : MacroAction.DEFEND_RUSH;
        }
        if (my.workers.isEmpty()) {
            return MacroAction.DEFEND_RUSH;
        }
        if (enemyNearBase && my.combatWithWorkers() + 1 < opp.combatWithWorkers()) {
            return MacroAction.DEFEND_RUSH;
        }
        if (hasRushPressure(my, opp, enemyNearBase, smallMap, opening)) {
            return MacroAction.DEFEND_RUSH;
        }
        if (!smallMap
                && !recentCombat
                && my.workers.size() >= 6
                && my.militaryCount() >= opp.militaryCount()
                && me.getResources() >= barracksType.cost
                && my.barracks.size() < 2) {
            return MacroAction.ECON_BOOM;
        }
        if (!smallMap
                && !opening
                && !my.barracks.isEmpty()
                && my.ranged.size() < Math.max(2, my.light.size() / 2)
                && my.workers.size() >= 5) {
            return MacroAction.RANGED_TRANSITION;
        }
        if (smallMap) {
            if (my.militaryCount() + 1 < opp.militaryCount()) {
                return MacroAction.DEFEND_RUSH;
            }
            return MacroAction.STANDARD_PRESSURE;
        }
        if (my.militaryCount() < opp.militaryCount()) {
            return MacroAction.DEFEND_RUSH;
        }
        if (!my.barracks.isEmpty() && my.ranged.size() < Math.max(1, my.light.size() / 2)) {
            return MacroAction.RANGED_TRANSITION;
        }
        return MacroAction.STANDARD_PRESSURE;
    }

    // ===========================
    // Feature extraction + policy
    // ===========================
    private float[] extractFeatures(
            GameState gs,
            int player,
            Counts my,
            Counts opp,
            boolean enemyNearBase,
            boolean recentCombat,
            float basePressure) {

        int enemy = 1 - player;
        float[] features = new float[FEATURE_COUNT];
        int i = 0;

        // Stable feature order for external PPO training/export.
        features[i++] = clamp01(gs.getTime() / 4000.0f);                                   // 0 time
        features[i++] = clamp01(gs.getPlayer(player).getResources() / 20.0f);              // 1 own res
        features[i++] = clamp01(gs.getPlayer(enemy).getResources() / 20.0f);               // 2 enemy res estimate
        features[i++] = clamp01(my.workers.size() / 10.0f);                                // 3 own workers
        features[i++] = clamp01(my.light.size() / 10.0f);                                  // 4 own light
        features[i++] = clamp01(my.ranged.size() / 8.0f);                                  // 5 own ranged
        features[i++] = clamp01(my.heavy.size() / 6.0f);                                   // 6 own heavy
        features[i++] = clamp01(opp.workers.size() / 10.0f);                               // 7 enemy workers
        features[i++] = clamp01(opp.light.size() / 10.0f);                                 // 8 enemy light
        features[i++] = clamp01(opp.ranged.size() / 8.0f);                                 // 9 enemy ranged
        features[i++] = clamp01(opp.heavy.size() / 6.0f);                                  // 10 enemy heavy
        features[i++] = my.bases.isEmpty() ? 0.0f : 1.0f;                                  // 11 own base alive
        features[i++] = my.barracks.isEmpty() ? 0.0f : 1.0f;                               // 12 own barracks alive
        features[i++] = opp.bases.isEmpty() ? 0.0f : 1.0f;                                 // 13 enemy base alive
        features[i++] = opp.barracks.isEmpty() ? 0.0f : 1.0f;                              // 14 enemy barracks alive
        features[i++] = normalizeSigned(my.workers.size() - opp.workers.size(), 10.0f);    // 15 worker diff
        features[i++] = normalizeSigned(my.militaryCount() - opp.militaryCount(), 10.0f);  // 16 military diff
        features[i++] = normalizeSigned(my.ranged.size() - opp.ranged.size(), 6.0f);       // 17 ranged diff
        features[i++] = normalizeSigned(my.light.size() - opp.light.size(), 6.0f);         // 18 light diff
        features[i++] = normalizeSigned(my.heavy.size() - opp.heavy.size(), 4.0f);         // 19 heavy diff
        features[i++] = basePressure;                                                       // 20 enemy-to-base pressure
        features[i++] = enemyNearBase ? 1.0f : 0.0f;                                        // 21 enemy near base
        features[i++] = currentMacroAction.ordinal() / (float) Math.max(1, ACTION_COUNT - 1); // 22 current macro id
        features[i++] = clamp01(strategyCooldown(gs.getTime()) / (float) POLICY_LOCK_TICKS);   // 23 change cooldown
        features[i++] = recentCombat ? 1.0f : 0.0f;                                         // 24 recent combat

        return features;
    }

    private PolicyOutput forwardPolicy(float[] features) {
        float[] hidden1 = dense(features, PPO_W1, PPO_B1);
        relu(hidden1);
        float[] hidden2 = dense(hidden1, PPO_W2, PPO_B2);
        relu(hidden2);
        float[] logits = dense(hidden2, PPO_W3, PPO_B3);
        float[] probabilities = softmax(logits);

        int bestIndex = 0;
        float best = probabilities[0];
        float second = Float.NEGATIVE_INFINITY;
        for (int i = 1; i < probabilities.length; i++) {
            float value = probabilities[i];
            if (value > best) {
                second = best;
                best = value;
                bestIndex = i;
            } else if (value > second) {
                second = value;
            }
        }
        if (second == Float.NEGATIVE_INFINITY) {
            second = 0.0f;
        }

        return new PolicyOutput(MACRO_ACTIONS[bestIndex], best, best - second);
    }

    private float[] dense(float[] input, float[][] weights, float[] bias) {
        float[] out = new float[bias.length];
        for (int j = 0; j < bias.length; j++) {
            float sum = bias[j];
            for (int i = 0; i < input.length; i++) {
                sum += input[i] * weights[i][j];
            }
            out[j] = sum;
        }
        return out;
    }

    private void relu(float[] values) {
        for (int i = 0; i < values.length; i++) {
            if (values[i] < 0.0f) {
                values[i] = 0.0f;
            }
        }
    }

    private float[] softmax(float[] logits) {
        float max = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > max) {
                max = logits[i];
            }
        }

        float sum = 0.0f;
        float[] out = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            out[i] = (float) Math.exp(logits[i] - max);
            sum += out[i];
        }
        if (sum <= 0.0f) {
            return out;
        }
        for (int i = 0; i < out.length; i++) {
            out[i] /= sum;
        }
        return out;
    }

    // ===========================
    // Deterministic production
    // ===========================
    private void handleBaseProduction(GameState gs, int player, Counts my, Counts opp, boolean enemyNearBase, boolean smallMap, boolean opening) {
        Player me = gs.getPlayer(player);
        int desiredWorkers = desiredWorkerCount(my, opp, enemyNearBase, smallMap, opening);

        for (Unit base : my.bases) {
            if (!isIdle(gs, base)) {
                continue;
            }
            if (my.workers.size() < desiredWorkers && me.getResources() >= workerType.cost) {
                train(base, workerType);
                // Conservative accounting to avoid over-issuing train orders in the same cycle.
                my.workers.add(new Unit(-1, workerType, 0, 0));
            }
        }
    }

    private int desiredWorkerCount(Counts my, Counts opp, boolean enemyNearBase, boolean smallMap, boolean opening) {
        if (opening) {
            switch (currentOpeningPlan) {
                case ANTI_WORKER_RUSH:
                    return smallMap ? 4 : 5;
                case ANTI_LIGHT_RUSH:
                    return smallMap ? 3 : 4;
                case ANTI_HEAVY_RUSH:
                    return smallMap ? 4 : 5;
                case LARGE_MAP_BOOM:
                    return smallMap ? 4 : 7;
                default:
                    return smallMap ? 4 : 6;
            }
        }

        int desired = smallMap ? (opening ? 4 : 5) : (opening ? 6 : 7);
        boolean rushPressure = hasRushPressure(my, opp, enemyNearBase, smallMap, opening);

        if (currentMacroAction.prefersRanged()) {
            desired += 1;
        }
        if (currentMacroAction.prefersEconomy()) {
            desired += smallMap ? 0 : 2;
        }
        if (currentMacroAction.prefersDefense()) {
            desired += 1;
        }
        if (currentMacroAction.prefersPressure() && smallMap) {
            desired -= 1;
        }
        if (enemyNearBase && my.militaryCount() <= 1) {
            desired = Math.max(desired, 6);
        }
        if (rushPressure) {
            desired = Math.min(desired, smallMap ? 4 : 5);
            if (!opp.heavy.isEmpty()) {
                desired = Math.min(desired, 4);
            }
        }

        return clampInt(desired, 2, smallMap ? 6 : 9);
    }

    private void handleBarracksProduction(
            GameState gs,
            int player,
            Counts my,
            Counts opp,
            boolean enemyNearBase,
            boolean smallMap,
            boolean opening) {

        Player me = gs.getPlayer(player);
        for (Unit barracks : my.barracks) {
            if (!isIdle(gs, barracks)) {
                continue;
            }

            UnitType chosen = chooseBarracksUnit(my, opp, enemyNearBase, smallMap, opening);
            if (chosen != null && me.getResources() >= chosen.cost) {
                train(barracks, chosen);
            }
        }
    }

    private UnitType chooseBarracksUnit(Counts my, Counts opp, boolean enemyNearBase, boolean smallMap, boolean opening) {
        if (opening) {
            switch (currentOpeningPlan) {
                case ANTI_WORKER_RUSH:
                case ANTI_LIGHT_RUSH:
                case ANTI_HEAVY_RUSH:
                case STANDARD_OPENING:
                    return lightType;
                case LARGE_MAP_BOOM:
                    if (!smallMap && my.light.size() >= 2 && my.ranged.size() < 1) {
                        return rangedType;
                    }
                    return lightType;
                default:
                    return lightType;
            }
        }

        boolean rushPressure = hasRushPressure(my, opp, enemyNearBase, smallMap, opening);
        if (enemyNearBase && my.combatWithWorkers() + 1 < opp.combatWithWorkers()) {
            return lightType;
        }
        if (rushPressure) {
            return lightType;
        }
        if (currentMacroAction.prefersDefense()) {
            return lightType;
        }
        if (smallMap) {
            return lightType;
        }
        if (currentMacroAction.prefersEconomy()) {
            if (my.light.size() < 4) {
                return lightType;
            }
            if (!opening && my.ranged.size() < Math.max(2, my.light.size() / 2)) {
                return rangedType;
            }
            return lightType;
        }
        if (currentMacroAction.prefersRanged() && !opening) {
            if (my.ranged.size() <= my.light.size()) {
                return rangedType;
            }
        }
        if (!opening && my.ranged.size() < Math.max(2, my.light.size() / 2) && my.light.size() >= 4) {
            return rangedType;
        }
        return lightType;
    }

    // ===========================
    // Worker manager (economy + defense + build timing)
    // ===========================
    private void handleWorkers(
            GameState gs,
            int player,
            Counts my,
            Counts opp,
            Unit myBase,
            Unit enemyBase,
            boolean enemyNearBase,
            boolean smallMap,
            boolean opening) {

        Player me = gs.getPlayer(player);
        int enemy = 1 - player;
        boolean defensiveOpening = opening
                && (currentOpeningPlan == OpeningPlan.ANTI_WORKER_RUSH
                || currentOpeningPlan == OpeningPlan.ANTI_LIGHT_RUSH
                || currentOpeningPlan == OpeningPlan.ANTI_HEAVY_RUSH);

        List<Unit> idleWorkers = new ArrayList<>();
        for (Unit w : my.workers) {
            if (isIdle(gs, w)) {
                idleWorkers.add(w);
            }
        }

        if (enemyNearBase && my.militaryCount() <= 1) {
            for (Unit w : idleWorkers) {
                Unit threat = nearestEnemy(gs.getPhysicalGameState(), w, enemy);
                if (threat != null) {
                    attack(w, threat);
                }
            }
            return;
        }

        if (shouldBuildBarracks(gs, player, my, smallMap, opening, enemyNearBase)
                && me.getResources() >= barracksType.cost
                && !idleWorkers.isEmpty()) {
            Unit builder = selectBuilder(idleWorkers, myBase);
            int[] site = chooseBarracksSite(gs.getPhysicalGameState(), myBase != null ? myBase : builder, enemy);
            if (builder != null && site != null) {
                build(builder, barracksType, site[0], site[1]);
                idleWorkers.remove(builder);
            }
        }

        int harvestersWanted = desiredHarvesterCount(my, opp, enemyNearBase, smallMap, opening);
        harvestersWanted = Math.min(harvestersWanted, Math.max(0, my.workers.size() - 1));

        List<Unit> resources = listResources(gs.getPhysicalGameState());
        int assigned = 0;
        for (Unit w : idleWorkers) {
            if (assigned >= harvestersWanted) {
                break;
            }
            Unit nearestRes = nearestResource(gs.getPhysicalGameState(), w, resources);
            if (nearestRes != null && myBase != null) {
                harvest(w, nearestRes, myBase);
                assigned++;
            }
        }

        for (Unit w : idleWorkers) {
            if (!isIdle(gs, w)) {
                continue;
            }

            if (currentMacroAction.prefersEconomy() && myBase != null) {
                Unit extraRes = nearestResource(gs.getPhysicalGameState(), w, resources);
                if (extraRes != null && !enemyNearBase) {
                    harvest(w, extraRes, myBase);
                    continue;
                }
            }

            Unit target = null;
            if (enemyNearBase || currentMacroAction.prefersDefense() || defensiveOpening) {
                target = nearestEnemy(gs.getPhysicalGameState(), w, enemy);
            } else if (opening && currentOpeningPlan == OpeningPlan.LARGE_MAP_BOOM && myBase != null) {
                Unit extraRes = nearestResource(gs.getPhysicalGameState(), w, resources);
                if (extraRes != null) {
                    harvest(w, extraRes, myBase);
                    continue;
                }
            } else if (my.militaryCount() < 3 || myBase == null) {
                target = nearestEnemyWithin(gs.getPhysicalGameState(), w, enemy, DEFENSE_RADIUS);
            } else if (enemyBase != null) {
                target = enemyBase;
            } else {
                target = nearestEnemy(gs.getPhysicalGameState(), w, enemy);
            }

            if (target != null) {
                attack(w, target);
            }
        }
    }

    private int desiredHarvesterCount(Counts my, Counts opp, boolean enemyNearBase, boolean smallMap, boolean opening) {
        if (opening) {
            switch (currentOpeningPlan) {
                case ANTI_WORKER_RUSH:
                case ANTI_LIGHT_RUSH:
                    return 1;
                case ANTI_HEAVY_RUSH:
                    return smallMap ? 2 : 3;
                case LARGE_MAP_BOOM:
                    return smallMap ? 2 : 4;
                default:
                    return smallMap ? 2 : 3;
            }
        }

        int harvesters = smallMap ? 2 : (opening ? 4 : 3);
        boolean rushPressure = hasRushPressure(my, opp, enemyNearBase, smallMap, opening);

        if (currentMacroAction.prefersEconomy()) {
            harvesters += smallMap ? 0 : 2;
        }
        if (currentMacroAction.prefersRanged()) {
            harvesters += 1;
        }
        if (currentMacroAction.prefersPressure()) {
            harvesters -= 1;
        }
        if (enemyNearBase) {
            harvesters = Math.max(1, harvesters - 1);
        }
        if (my.barracks.size() >= 2 && !smallMap) {
            harvesters = Math.min(harvesters + 1, 5);
        }
        if (rushPressure) {
            harvesters = Math.min(harvesters, 2);
        }

        return clampInt(harvesters, 1, smallMap ? 3 : 5);
    }

    private boolean shouldBuildBarracks(GameState gs, int player, Counts my, boolean smallMap, boolean opening, boolean enemyNearBase) {
        if (my.barracks.isEmpty()) {
            return !my.workers.isEmpty();
        }
        if (smallMap || opening || enemyNearBase) {
            return false;
        }
        if (my.barracks.size() >= 2) {
            return false;
        }

        Player me = gs.getPlayer(player);
        if (currentMacroAction.prefersDefense()) {
            return my.workers.size() >= 6 && my.militaryCount() >= 6 && me.getResources() >= barracksType.cost;
        }
        if (currentMacroAction.prefersEconomy()) {
            return my.workers.size() >= 5 && me.getResources() >= barracksType.cost;
        }
        return my.workers.size() >= 5 && my.militaryCount() >= 4 && me.getResources() >= barracksType.cost;
    }

    // ===========================
    // Military execution
    // ===========================
    private void handleMilitary(
            GameState gs,
            int enemy,
            Counts my,
            Counts opp,
            Unit myBase,
            Unit enemyBase,
            boolean enemyNearBase,
            boolean opening) {

        List<Unit> army = new ArrayList<>();
        army.addAll(my.light);
        army.addAll(my.ranged);
        army.addAll(my.heavy);
        boolean defensiveOpening = opening && isOpeningPlanDefensive();

        boolean urgentDefense = enemyNearBase
                || defensiveOpening
                || currentMacroAction.prefersDefense()
                || (currentMacroAction.prefersEconomy() && my.militaryCount() < 5);

        for (Unit unit : army) {
            if (!isIdle(gs, unit)) {
                continue;
            }

            Unit target = selectCombatTarget(gs.getPhysicalGameState(), unit, enemy, myBase, urgentDefense, opp);
            if (target == null && enemyBase != null && !defensiveOpening) {
                if (currentMacroAction.attacksWhenStable()
                        || (currentMacroAction.prefersEconomy() && my.militaryCount() >= 5)) {
                    target = enemyBase;
                }
            }
            if (target != null) {
                attack(unit, target);
            }
        }
    }

    private Unit selectCombatTarget(PhysicalGameState pgs, Unit attacker, int enemy, Unit myBase, boolean urgentDefense, Counts opp) {
        List<Unit> enemies = new ArrayList<>();
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == enemy) {
                enemies.add(u);
            }
        }
        if (enemies.isEmpty()) {
            return null;
        }

        enemies.sort(Comparator.comparingInt(e -> targetScore(attacker, e, myBase, urgentDefense, opp)));
        return enemies.get(0);
    }

    private int targetScore(Unit attacker, Unit target, Unit myBase, boolean urgentDefense, Counts opp) {
        int d = manhattan(attacker, target);
        int typeBias;

        if (target.getType() == workerType) {
            typeBias = (opp.workers.size() <= 2) ? 1 : 5;
        } else if (target.getType() == lightType || target.getType() == rangedType || target.getType() == heavyType) {
            typeBias = 3;
        } else if (target.getType() == barracksType) {
            typeBias = 7;
        } else if (target.getType() == baseType) {
            typeBias = 9;
        } else {
            typeBias = 6;
        }

        if (urgentDefense && target.getType().canAttack) {
            typeBias -= 2;
        }
        if (currentMacroAction.prefersPressure() && target.getType() == baseType) {
            typeBias -= 2;
        }
        if (currentMacroAction.prefersRanged() && target.getType() == rangedType) {
            typeBias -= 1;
        }
        if (currentMacroAction.prefersEconomy() && target.getType().canAttack) {
            typeBias -= 2;
        }

        int baseProximity = 0;
        if (myBase != null) {
            baseProximity = manhattan(myBase, target);
        }

        return d * 3 + typeBias + (urgentDefense ? baseProximity * 2 : 0);
    }

    // ===========================
    // Tactical MCTS gating
    // ===========================
    private boolean shouldUseMCTS(
            GameState gs,
            Counts my,
            Counts opp,
            boolean opening,
            boolean enemyNearBase,
            boolean smallMap,
            boolean recentCombat) {

        if (my.bases.isEmpty() || my.workers.size() < 1) {
            return false;
        }
        boolean criticalOpeningDefense = opening
                && isOpeningPlanDefensive()
                && enemyNearBase
                && my.combatWithWorkers() >= Math.max(2, opp.combatWithWorkers() - 1);

        if (enemyNearBase && my.militaryCount() <= 1 && !criticalOpeningDefense) {
            return false;
        }

        int myArmy = my.militaryCount();
        int oppArmy = opp.militaryCount();

        boolean baseDefenseFight = enemyNearBase && myArmy >= 2 && opp.combatWithWorkers() >= 2;
        boolean balancedSkirmish = recentCombat && myArmy >= 3 && oppArmy >= 3 && Math.abs(myArmy - oppArmy) <= 3;
        boolean denseFight = recentCombat && myArmy + oppArmy >= 8 && Math.abs(myArmy - oppArmy) <= 4;
        boolean openingDefenseFight = opening && enemyNearBase && my.combatWithWorkers() >= opp.combatWithWorkers() - 1;

        if (opening && !baseDefenseFight && !openingDefenseFight && !criticalOpeningDefense) {
            return false;
        }
        if (smallMap && gs.getTime() < SMALL_MAP_OPENING_TICKS && !baseDefenseFight && !balancedSkirmish && !criticalOpeningDefense) {
            return false;
        }
        return baseDefenseFight || openingDefenseFight || balancedSkirmish || denseFight || criticalOpeningDefense;
    }

    // ===========================
    // Geometry / state helpers
    // ===========================
    private boolean isIdle(GameState gs, Unit u) {
        return gs.getActionAssignment(u) == null;
    }

    private boolean isEnemyNearBase(PhysicalGameState pgs, Counts my, int enemy, int radius) {
        if (my.bases.isEmpty()) {
            return false;
        }
        Unit base = my.bases.get(0);
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != enemy) {
                continue;
            }
            if (!u.getType().canAttack && u.getType() != workerType) {
                continue;
            }
            if (manhattan(base, u) <= radius) {
                return true;
            }
        }
        return false;
    }

    private float basePressureIndicator(PhysicalGameState pgs, Counts my, int enemy) {
        if (my.bases.isEmpty()) {
            return 1.0f;
        }
        Unit base = my.bases.get(0);
        int best = Integer.MAX_VALUE;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != enemy) {
                continue;
            }
            if (!u.getType().canAttack && u.getType() != workerType) {
                continue;
            }
            int d = manhattan(base, u);
            if (d < best) {
                best = d;
            }
        }
        if (best == Integer.MAX_VALUE) {
            return 0.0f;
        }
        float dangerDistance = Math.max(6.0f, (pgs.getWidth() + pgs.getHeight()) / 3.0f);
        return 1.0f - clamp01(best / dangerDistance);
    }

    private boolean isCombatHappening(PhysicalGameState pgs, Counts my, int enemy) {
        for (Unit ally : my.mobileUnits()) {
            for (Unit foe : pgs.getUnits()) {
                if (foe.getPlayer() != enemy) {
                    continue;
                }
                if (!foe.getType().canAttack && foe.getType() != workerType) {
                    continue;
                }
                int engagementDistance = Math.max(ally.getAttackRange(), foe.getAttackRange()) + 2;
                if (manhattan(ally, foe) <= engagementDistance) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean hasRushPressure(Counts my, Counts opp, boolean enemyNearBase, boolean smallMap, boolean opening) {
        if (enemyNearBase) {
            return true;
        }
        if (opp.barracks.isEmpty()) {
            return false;
        }
        if (!opp.heavy.isEmpty()) {
            return true;
        }
        if (opening && opp.militaryCount() > 0) {
            return true;
        }
        if (smallMap && opp.militaryCount() > my.militaryCount()) {
            return true;
        }
        return opp.militaryCount() > my.militaryCount() + 1;
    }

    private boolean isOpeningPlanDefensive() {
        return currentOpeningPlan == OpeningPlan.ANTI_WORKER_RUSH
                || currentOpeningPlan == OpeningPlan.ANTI_LIGHT_RUSH
                || currentOpeningPlan == OpeningPlan.ANTI_HEAVY_RUSH;
    }

    private void updateOpeningTelemetry(int now, Counts my, Counts opp, boolean opening) {
        if (openingStartWorkerCount < 0) {
            openingStartWorkerCount = my.workers.size();
        }
        if (firstEnemyBarracksSeenTime < 0 && !opp.barracks.isEmpty()) {
            firstEnemyBarracksSeenTime = now;
        }
        if (firstEnemyMilitarySeenTime < 0 && opp.militaryCount() > 0) {
            firstEnemyMilitarySeenTime = now;
        }
        if (firstOwnBarracksSeenTime < 0 && !my.barracks.isEmpty()) {
            firstOwnBarracksSeenTime = now;
        }
        if (firstDefenderSeenTime < 0 && my.militaryCount() > 0) {
            firstDefenderSeenTime = now;
        }
        if (opening) {
            openingWorkerLosses = Math.max(openingWorkerLosses, Math.max(0, openingStartWorkerCount - my.workers.size()));
        }
    }

    private boolean isOpeningPhase(GameState gs, boolean smallMap) {
        return gs.getTime() < (smallMap ? SMALL_MAP_OPENING_TICKS : LARGE_MAP_OPENING_TICKS);
    }

    private int[] chooseBarracksSite(PhysicalGameState pgs, Unit anchor, int enemy) {
        if (anchor == null) {
            return null;
        }

        int[][] ring = {
                {1, 0}, {-1, 0}, {0, 1}, {0, -1},
                {2, 0}, {-2, 0}, {0, 2}, {0, -2},
                {1, 1}, {-1, 1}, {1, -1}, {-1, -1}
        };

        int bestScore = Integer.MIN_VALUE;
        int[] best = null;

        for (int[] d : ring) {
            int x = anchor.getX() + d[0];
            int y = anchor.getY() + d[1];
            if (x < 0 || y < 0 || x >= pgs.getWidth() || y >= pgs.getHeight()) {
                continue;
            }
            if (pgs.getTerrain(x, y) != PhysicalGameState.TERRAIN_NONE) {
                continue;
            }
            if (pgs.getUnitAt(x, y) != null) {
                continue;
            }

            int baseDist = Math.abs(anchor.getX() - x) + Math.abs(anchor.getY() - y);
            int enemyDist = nearestEnemyDistanceFromCell(pgs, x, y, enemy);
            int score = enemyDist * 2 - baseDist;

            if (score > bestScore) {
                bestScore = score;
                best = new int[] {x, y};
            }
        }
        return best;
    }

    private Unit selectBuilder(List<Unit> workers, Unit base) {
        if (workers.isEmpty()) {
            return null;
        }
        if (base == null) {
            return workers.get(0);
        }

        Unit best = null;
        int bestD = Integer.MAX_VALUE;
        for (Unit w : workers) {
            int d = manhattan(w, base);
            if (d < bestD) {
                bestD = d;
                best = w;
            }
        }
        return best;
    }

    private Unit nearestEnemy(PhysicalGameState pgs, Unit from, int enemy) {
        Unit best = null;
        int bestD = Integer.MAX_VALUE;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != enemy) {
                continue;
            }
            int d = manhattan(from, u);
            if (d < bestD) {
                bestD = d;
                best = u;
            }
        }
        return best;
    }

    private Unit nearestEnemyWithin(PhysicalGameState pgs, Unit from, int enemy, int maxDistance) {
        Unit best = null;
        int bestD = Integer.MAX_VALUE;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != enemy) {
                continue;
            }
            int d = manhattan(from, u);
            if (d <= maxDistance && d < bestD) {
                bestD = d;
                best = u;
            }
        }
        return best;
    }

    private int nearestEnemyDistanceFromCell(PhysicalGameState pgs, int x, int y, int enemy) {
        int best = Integer.MAX_VALUE;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != enemy) {
                continue;
            }
            int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
            if (d < best) {
                best = d;
            }
        }
        return best;
    }

    private Unit nearestResource(PhysicalGameState pgs, Unit from, List<Unit> resources) {
        Unit best = null;
        int bestD = Integer.MAX_VALUE;
        for (Unit r : resources) {
            int d = manhattan(from, r);
            if (d < bestD) {
                bestD = d;
                best = r;
            }
        }
        return best;
    }

    private List<Unit> listResources(PhysicalGameState pgs) {
        List<Unit> out = new ArrayList<>();
        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) {
                out.add(u);
            }
        }
        return out;
    }

    private int manhattan(Unit a, Unit b) {
        return Math.abs(a.getX() - b.getX()) + Math.abs(a.getY() - b.getY());
    }

    private Counts countUnits(PhysicalGameState pgs, int player) {
        Counts c = new Counts();
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != player) {
                continue;
            }
            if (u.getType() == workerType) {
                c.workers.add(u);
            } else if (u.getType() == baseType) {
                c.bases.add(u);
            } else if (u.getType() == barracksType) {
                c.barracks.add(u);
            } else if (u.getType() == lightType) {
                c.light.add(u);
            } else if (u.getType() == rangedType) {
                c.ranged.add(u);
            } else if (u.getType() == heavyType) {
                c.heavy.add(u);
            }
        }
        return c;
    }

    private float clamp01(float value) {
        if (value < 0.0f) {
            return 0.0f;
        }
        if (value > 1.0f) {
            return 1.0f;
        }
        return value;
    }

    private float normalizeSigned(int value, float normalizer) {
        return Math.max(-1.0f, Math.min(1.0f, value / normalizer));
    }

    private int clampInt(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private int strategyCooldown(int now) {
        if (lastStrategyChangeTime < 0) {
            return POLICY_LOCK_TICKS;
        }
        return now - lastStrategyChangeTime;
    }

    private static final class Counts {
        final List<Unit> workers = new ArrayList<>();
        final List<Unit> bases = new ArrayList<>();
        final List<Unit> barracks = new ArrayList<>();
        final List<Unit> light = new ArrayList<>();
        final List<Unit> ranged = new ArrayList<>();
        final List<Unit> heavy = new ArrayList<>();

        int militaryCount() {
            return light.size() + ranged.size() + heavy.size();
        }

        int combatWithWorkers() {
            return militaryCount() + Math.min(2, workers.size());
        }

        List<Unit> mobileUnits() {
            List<Unit> units = new ArrayList<>(workers.size() + light.size() + ranged.size() + heavy.size());
            units.addAll(workers);
            units.addAll(light);
            units.addAll(ranged);
            units.addAll(heavy);
            return units;
        }
    }

    private static final class PolicyOutput {
        final MacroAction action;
        final float confidence;
        final float margin;

        PolicyOutput(MacroAction action, float confidence, float margin) {
            this.action = action;
            this.confidence = confidence;
            this.margin = margin;
        }
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}
