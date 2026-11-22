# 보안 정책 / Security policy

보안 취약점을 발견하셨다면 **공개 이슈 대신** 이 레포의 **Security → Report a vulnerability**(비공개 취약점 신고)로 알려 주세요. 확인 후 답변드리고, 고친 버전을 릴리스하면서 함께 안내합니다.

If you find a security issue, please **do not open a public issue**. Use **Security → Report a vulnerability** in this repository to report it privately. We will reply after checking and announce the fix with a release.

masklens 는 네트워크 서버를 열지 않는 로컬 CLI 이며, Hub ID 를 넘길 때만 Hugging Face Hub 에 접속합니다. trust_remote_code 는 쓰지 않습니다. / masklens is a local CLI that opens no network server; it contacts the Hugging Face Hub only when you pass a Hub ID. It does not use trust_remote_code.
