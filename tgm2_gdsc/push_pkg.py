import pexpect, sys, os, tarfile

SRC = "/store/zkyang/tgm2_gdsc/pkg"
TAR = "/store/zkyang/tgm2_gdsc/tgm2_pkg.tar.gz"
PW = "Ysu2024!"
DST_DIR = "/storu/ysu/nfcore/wenjie"

with tarfile.open(TAR, "w:gz") as t:
    for root, _, files in os.walk(SRC):
        for f in files:
            p = os.path.join(root, f)
            t.add(p, arcname=os.path.relpath(p, SRC))
print("tar 大小", os.path.getsize(TAR))


def run(cmd, timeout=300):
    c = pexpect.spawn("/bin/bash", ["-c", cmd], timeout=timeout, encoding="utf-8")
    c.logfile_read = sys.stdout
    try:
        while True:
            i = c.expect([r"[Pp]assword:", r"passphrase", r"\(yes/no",
                          pexpect.EOF, pexpect.TIMEOUT])
            if i in (0, 1):
                c.sendline(PW)
            elif i == 2:
                c.sendline("yes")
            else:
                break
    except Exception as e:
        print("\nERR:", e)
    c.close()
    return c.exitstatus


SSHO = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
print("\n=== 1. scp tar ===")
run(f'scp {SSHO} "{TAR}" ysu@node4:{DST_DIR}/tgm2_pkg.tar.gz')

print("\n=== 2. 远端解包 ===")
remote = (f"cd {DST_DIR} && tar xzf tgm2_pkg.tar.gz && rm -f tgm2_pkg.tar.gz && "
          f"ls -R {DST_DIR} | head -60 && echo '--- 磁盘 ---' && du -sh {DST_DIR}")
run(f'ssh {SSHO} ysu@node4 "{remote}"')
