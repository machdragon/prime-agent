import { spawnSync } from "node:child_process";
import { mkdirSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { getBundledSkillsDir } from "../src/config.js";
import { loadSkillsFromDir, type PythonSkillRuntimeInfo } from "../src/core/skills.js";
import { IpythonKernelProvisioner } from "../src/core/tools/ipython.js";

const skillDir = join(getBundledSkillsDir(), "goal-graph");

function bundledGoalGraphSkill(): PythonSkillRuntimeInfo {
	return {
		name: "goal-graph",
		importName: "goal_graph",
		packagePath: skillDir,
		pyprojectPath: join(skillDir, "pyproject.toml"),
	};
}

describe("goal-graph skill logic", () => {
	it("is discovered as a bundled python skill", () => {
		const { skills, diagnostics } = loadSkillsFromDir({ dir: getBundledSkillsDir(), source: "builtin" });
		const skill = skills.find((candidate) => candidate.name === "goal-graph");
		expect(diagnostics.filter((diagnostic) => diagnostic.path?.includes("goal-graph"))).toEqual([]);
		expect(skill?.python).toMatchObject({
			importName: "goal_graph",
			packagePath: skillDir,
			pyprojectPath: join(skillDir, "pyproject.toml"),
		});
	});

	it("passes its Python suite", () => {
		const result = spawnSync("python3", ["-m", "unittest", "discover", "-s", "test"], {
			cwd: skillDir,
			encoding: "utf8",
			env: { ...process.env, PYTHONPATH: join(skillDir, "src"), PYTHONDONTWRITEBYTECODE: "1" },
		});
		expect(result.error).toBeUndefined();
		expect(`${result.stdout}${result.stderr}`).toContain("OK");
		expect(result.status).toBe(0);
	});
});

describe("goal-graph skill in a live kernel", { tags: ["kernel-heavy"] }, () => {
	let tempDir: string;
	let provisioner: IpythonKernelProvisioner | undefined;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-goal-graph-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });
	});

	afterEach(async () => {
		await provisioner?.dispose();
		provisioner = undefined;
		rmSync(tempDir, { recursive: true, force: true });
	});

	it("decomposes and runs a graph, then resumes it from the store", async () => {
		provisioner = new IpythonKernelProvisioner(tempDir, { pythonSkills: [bundledGoalGraphSkill()] });
		const manager = await provisioner.ensure();

		const ran = await manager.execute(`
import json, os
os.environ["RLM_GOAL_GRAPH_DIR"] = ${JSON.stringify(join(tempDir, "graphs"))}
from goal_graph import Graph, Node, Expand, Done, register

@register
def fan(ctx):
    return Expand([Node(intent=f"leaf {i}", fn="leaf", args={"i": i}) for i in range(3)])

@register
def leaf(ctx):
    return Done(ctx.args["i"] * 2)

_graph = Graph.open("kernel-demo")
_root = _graph.add(Node(intent="fan out", fn="fan"))
_report = await _graph.run()
print(json.dumps({"stopped": _report.stopped, "nodes": len(_graph), "result": sorted(_graph.get(_root.id).result)}))
`);
		expect(ran.status).toBe("ok");
		expect(JSON.parse(ran.stdout.trim())).toEqual({ stopped: "complete", nodes: 4, result: [0, 2, 4] });

		const reloaded = await manager.execute(`
_again = Graph.open("kernel-demo")
_pending = _again.add(Node(intent="needs a model", prompt="decide the next step"))
_second = await _again.run()
print(_second.stopped, len(_again), _second.pending_dispatch == (_pending.id,), _again.path)
`);
		expect(reloaded.status).toBe("ok");
		// The store resolves symlinks, which on macOS turns /var into /private/var.
		expect(reloaded.stdout.trim()).toBe(
			`pending_dispatch 5 True ${join(realpathSync(tempDir), "graphs", "kernel-demo.json")}`,
		);
	});
});
