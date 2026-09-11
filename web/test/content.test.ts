import { describe, expect, it } from "vitest";
import { validateContent } from "../scripts/validate-content.ts";
import { commandGroups, commands } from "../src/data/commands.ts";
import { pipeline, reliabilityClaims, REPOSITORY_BRANCH, REPOSITORY_URL, sourceQuickstart } from "../src/data/site.ts";
import {
  XLAB_CONTROL_COMMANDS,
  XLAB_PRODUCT_COMMANDS,
} from "../../packages/coding-agent/src/core/xlab/product-commands.ts";

describe("repository-backed content", () => {
  it("matches canonical commands and public skill manifests", async () => {
    await expect(validateContent()).resolves.toEqual([]);
  });

  it("contains the complete canonical command catalog", () => {
    const product = commands.filter(({ kind }) => kind === "product");
    const control = commands.filter(({ kind }) => kind === "control");
    expect(commands).toHaveLength(25);
    expect(product).toHaveLength(14);
    expect(control).toHaveLength(11);
    expect(new Set(product.map(({ task }) => task))).toEqual(new Set(XLAB_PRODUCT_COMMANDS.map(({ task }) => task)));
    expect(new Set(control.map(({ task }) => task))).toEqual(new Set(XLAB_CONTROL_COMMANDS.map(({ task }) => task)));
  });

  it("keeps command identifiers, anchors, and groups valid", () => {
    const groups = new Set(commandGroups.map(({ id }) => id));
    expect(new Set(commands.map(({ task }) => task)).size).toBe(commands.length);
    expect(new Set(commands.map(({ anchor }) => anchor)).size).toBe(commands.length);
    expect(commands.every(({ group }) => groups.has(group))).toBe(true);
    expect(new Set(pipeline.map(({ skillName }) => skillName)).size).toBe(pipeline.length);
  });

  it("keeps the homepage reliability mechanisms repository-backed", () => {
    const featured = ["Durable runs", "Typed artifacts", "Acceptance gates"];
    expect(featured.every((title) => reliabilityClaims.some((claim) => claim.title === title))).toBe(true);
    expect(reliabilityClaims.every((claim) => claim.source.path.length > 0)).toBe(true);
  });

  it("uses the source quickstart", () => {
    expect(sourceQuickstart).toEqual(["npm install --ignore-scripts", "./pi-test.sh"]);
  });

  it("links to the public XLab repository on its durable branch", () => {
    expect(REPOSITORY_URL).toBe("https://github.com/daqige/XLab");
    expect(REPOSITORY_BRANCH).toBe("main");
  });
});
